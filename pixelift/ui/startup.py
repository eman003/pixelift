"""The startup screen: Pixelift's logo, a status line and a quiet loading indicator.

Shown in the main window while it builds its workspace and the processing
device is detected, then dissolved into the workspace (see MainWindow). The
animation is a transition, never a delay: the window switches as soon as it is
ready and the logo has settled (about 0.65 s into the screen).

Timeline (ms after the screen first appears):

* 100–550    the logo fades in and grows from 90 % to its full size (ease-out);
* 450–850    a soft brand glow rises behind it, then settles (ease-in-out);
* 600–1150   a single light sweep crosses the logo tile;
* afterwards a very slow ambient pulse of the glow, only while still waiting.

Everything is drawn with GTK's own snapshot API from the existing app icon.
With animations turned off (GNOME's reduced-motion setting) the screen is
static.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from gi.repository import Adw, Gdk, Graphene, Gsk, Gtk, Pango

from pixelift import APP_ID, APP_NAME

LOGO_SIZE = 112
GLOW_RADIUS = 100  # px from the logo's centre
BRAND = (0x6D / 255, 0x4D / 255, 0xFF / 255)  # the logo tile's violet
INTRO_MS = 1150
LOGO_SETTLED_MS = 650  # the logo is fully in: the screen may give way
PULSE_MS = 2600
# The icon's tile inside its 128 px canvas (see the SVG): sweep is clipped to it.
TILE_INSET, TILE_RADIUS = 8 / 128, 26 / 128

STATUS_STARTING = "Starting Pixelift"
STATUS_DEVICE = "Checking processing device"
STATUS_READY = "Ready"


def animations_enabled() -> bool:
    settings = Gtk.Settings.get_default()
    return settings is None or bool(settings.props.gtk_enable_animations)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _ease_out(t: float) -> float:
    return 1 - (1 - _clamp(t)) ** 3


def _ease_in_out(t: float) -> float:
    t = _clamp(t)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _rgba(r: float, g: float, b: float, a: float) -> Gdk.RGBA:
    color = Gdk.RGBA()
    color.red, color.green, color.blue, color.alpha = r, g, b, a
    return color


def _stop(offset: float, color: Gdk.RGBA) -> Gsk.ColorStop:
    stop = Gsk.ColorStop()
    stop.offset = offset
    stop.color = color
    return stop


def _rounded(rect: Graphene.Rect, radius: float) -> Gsk.RoundedRect:
    # Keep the struct referenced: chaining ``Gsk.RoundedRect().init_from_rect()``
    # hands GTK a struct Python has already freed.
    rounded = Gsk.RoundedRect()
    rounded.init_from_rect(rect, radius)
    return rounded


class LogoMark(Gtk.Widget):
    """The app icon with its entrance, glow and light sweep."""

    __gtype_name__ = "PixeliftStartupLogo"

    def __init__(self) -> None:
        super().__init__(halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
        side = 2 * GLOW_RADIUS
        self.set_size_request(side, side)
        self.time_ms = 0.0  # position on the intro timeline
        self.pulse = 0.0  # 0..1, the ambient pulse after the intro
        self._paintable: Gdk.Paintable | None = None
        self.update_property([Gtk.AccessibleProperty.LABEL], [f"{APP_NAME} logo"])

    def _logo(self) -> Gdk.Paintable | None:
        if self._paintable is None:
            display = self.get_display()
            theme = Gtk.IconTheme.get_for_display(display)
            self._paintable = theme.lookup_icon(
                APP_ID, None, LOGO_SIZE, self.get_scale_factor(), Gtk.TextDirection.NONE, 0
            )
        return self._paintable

    # --- the timeline --------------------------------------------------------
    def entrance(self) -> float:
        return _ease_out((self.time_ms - 100) / 450)

    def glow(self) -> float:
        t = self.time_ms
        rise = _ease_in_out((t - 450) / 400)  # 0 → 1 by 850 ms
        settle = _ease_in_out((t - 850) / 300)  # then down to 0.6
        level = rise * (1 - 0.4 * settle)
        if t >= INTRO_MS:
            level = 0.6 + 0.12 * math.sin(self.pulse * 2 * math.pi)
        return level

    def sweep(self) -> float | None:
        """Position of the light band across the tile (0..1), None outside it."""
        t = (self.time_ms - 600) / 550
        return None if t <= 0 or t >= 1 else _ease_in_out(t)

    # --- drawing -------------------------------------------------------------
    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        width, height = self.get_width(), self.get_height()
        cx, cy = width / 2, height / 2
        entrance = self.entrance()

        glow = self.glow() * entrance
        if glow > 0.01:
            r, g, b = BRAND
            center = Graphene.Point().init(cx, cy)
            snapshot.append_radial_gradient(
                Graphene.Rect().init(0, 0, width, height),
                center,
                GLOW_RADIUS,
                GLOW_RADIUS,
                0.0,
                1.0,
                [
                    _stop(0.0, _rgba(r, g, b, 0.30 * glow)),
                    _stop(0.45, _rgba(r, g, b, 0.12 * glow)),
                    _stop(1.0, _rgba(r, g, b, 0.0)),
                ],
            )

        logo = self._logo()
        if logo is None or entrance <= 0:
            return
        scale = 0.9 + 0.1 * entrance
        snapshot.save()
        snapshot.translate(Graphene.Point().init(cx, cy))
        snapshot.scale(scale, scale)
        snapshot.translate(Graphene.Point().init(-LOGO_SIZE / 2, -LOGO_SIZE / 2))
        snapshot.push_opacity(entrance)
        logo.snapshot(snapshot, LOGO_SIZE, LOGO_SIZE)

        position = self.sweep()
        if position is not None:
            inset, side = LOGO_SIZE * TILE_INSET, LOGO_SIZE * (1 - 2 * TILE_INSET)
            tile = Graphene.Rect().init(inset, inset, side, side)
            snapshot.push_rounded_clip(_rounded(tile, LOGO_SIZE * TILE_RADIUS))
            # A soft diagonal band of light, top-left to bottom-right.
            band = 0.18
            centre = -band + position * (1 + 2 * band)
            stops = [
                (centre - band, 0.0),
                (centre, 0.32),
                (centre + band, 0.0),
            ]
            snapshot.append_linear_gradient(
                tile,
                Graphene.Point().init(inset, inset),
                Graphene.Point().init(inset + side, inset + side),
                [_stop(_clamp(offset), _rgba(1, 1, 1, alpha)) for offset, alpha in stops],
            )
            snapshot.pop()

        snapshot.pop()
        snapshot.restore()


class LoadingDots(Gtk.Widget):
    """Three small dots with a gentle travelling fade (no spinner)."""

    __gtype_name__ = "PixeliftStartupDots"
    COUNT, SIZE, GAP = 3, 6, 8

    def __init__(self) -> None:
        super().__init__(halign=Gtk.Align.CENTER)
        self.add_css_class("startup-dots")
        self.set_size_request(self.COUNT * self.SIZE + (self.COUNT - 1) * self.GAP, self.SIZE)
        self.phase = 0.0  # 0..1
        self.moving = True
        self.update_property([Gtk.AccessibleProperty.LABEL], ["Loading"])

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        color = self.get_color()
        y = (self.get_height() - self.SIZE) / 2
        for index in range(self.COUNT):
            if self.moving:
                wave = 0.5 + 0.5 * math.cos(2 * math.pi * (self.phase - index * 0.16))
                alpha = 0.25 + 0.75 * wave
            else:
                alpha = 0.6
            x = index * (self.SIZE + self.GAP)
            rect = Graphene.Rect().init(x, y, self.SIZE, self.SIZE)
            snapshot.push_rounded_clip(_rounded(rect, self.SIZE / 2))
            snapshot.append_color(
                _rgba(color.red, color.green, color.blue, color.alpha * alpha), rect
            )
            snapshot.pop()


class StartupScreen(Adw.Bin):
    """Logo, name, status and loading dots — or a clean error with Try Again / Exit."""

    def __init__(self, on_retry: Callable[[], None], on_exit: Callable[[], None]) -> None:
        super().__init__()
        self.add_css_class("startup-screen")
        self.logo_settled = False
        self._settled_callbacks: list[Callable[[], None]] = []
        self._animations: list[Adw.Animation] = []
        self._stopped = False

        # A bare, flat header keeps the window movable and closable meanwhile.
        header = Adw.HeaderBar(show_title=False)
        header.add_css_class("flat")
        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(header)

        column = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
            margin_bottom=48,  # optically centred: the header sits above
        )
        self.logo = LogoMark()
        column.append(self.logo)
        self.title = Gtk.Label(label=APP_NAME, margin_top=4)
        self.title.add_css_class("title-1")
        self.title.add_css_class("startup-title")
        column.append(self.title)

        self.status = Gtk.Label(label=STATUS_STARTING, margin_top=6)
        self.status.add_css_class("dim-label")
        self.status.add_css_class("startup-status")
        self.status.set_accessible_role(Gtk.AccessibleRole.STATUS)
        column.append(self.status)
        self.dots = LoadingDots()
        self.dots.set_margin_top(22)
        column.append(self.dots)

        column.append(self._build_error(on_retry, on_exit))
        toolbar.set_content(column)
        self.set_child(toolbar)
        self.title.set_opacity(0)
        self.status.set_opacity(0)
        self.dots.set_opacity(0)
        self.connect("map", lambda _w: self._start())

    def _build_error(self, on_retry: Callable[[], None], on_exit: Callable[[], None]) -> Gtk.Widget:
        self.error_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_top=8, visible=False
        )
        message = Gtk.Label(
            label="Something prevented Pixelift from starting correctly.",
            wrap=True,
            justify=Gtk.Justification.CENTER,
            max_width_chars=44,
        )
        message.add_css_class("dim-label")
        self.error_box.append(message)
        buttons = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER, margin_top=12)
        self.retry_btn = Gtk.Button(label="Try Again")
        self.retry_btn.add_css_class("pill")
        self.retry_btn.add_css_class("suggested-action")
        self.retry_btn.connect("clicked", lambda _b: on_retry())
        exit_btn = Gtk.Button(label="Exit")
        exit_btn.add_css_class("pill")
        exit_btn.connect("clicked", lambda _b: on_exit())
        buttons.append(self.retry_btn)
        buttons.append(exit_btn)
        self.error_box.append(buttons)
        self.details = Gtk.Label(
            xalign=0, selectable=True, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR
        )
        self.details.add_css_class("monospace")
        self.details.add_css_class("caption")
        scroller = Gtk.ScrolledWindow(
            child=self.details,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            propagate_natural_height=True,
            max_content_height=180,
            min_content_width=420,
        )
        scroller.add_css_class("card")
        scroller.add_css_class("startup-details")
        expander = Gtk.Expander(label="View details", halign=Gtk.Align.CENTER, margin_top=6)
        expander.set_child(scroller)
        self.error_box.append(expander)
        return self.error_box

    # --- animation -----------------------------------------------------------
    def _start(self) -> None:
        if self._animations or self._stopped:
            return
        if not animations_enabled():
            # Reduced motion: the finished, static screen.
            self.logo.time_ms = INTRO_MS
            self.dots.moving = False
            for widget in (self.title, self.status, self.dots):
                widget.set_opacity(1)
            self._settle()
            return

        def tick(value: float) -> None:
            self.logo.time_ms = value
            self.logo.queue_draw()
            # The name follows the logo, the status and dots a little later.
            self.title.set_opacity(_ease_out((value - 250) / 400))
            self.status.set_opacity(_ease_out((value - 400) / 400))
            self.dots.set_opacity(_ease_out((value - 500) / 400))
            if value >= LOGO_SETTLED_MS:
                self._settle()

        intro = Adw.TimedAnimation(
            widget=self,
            value_from=0,
            value_to=INTRO_MS,
            duration=INTRO_MS,
            easing=Adw.Easing.LINEAR,
            target=Adw.CallbackAnimationTarget.new(tick),
        )
        intro.connect("done", lambda _a: self._after_intro())
        self._animations.append(intro)

        def dots_tick(value: float) -> None:
            self.dots.phase = value
            self.dots.queue_draw()

        dots = Adw.TimedAnimation(
            widget=self.dots,
            value_from=0,
            value_to=1,
            duration=1200,
            repeat_count=0,  # until the screen goes away
            easing=Adw.Easing.LINEAR,
            target=Adw.CallbackAnimationTarget.new(dots_tick),
        )
        self._animations.append(dots)
        intro.play()
        dots.play()

    def _after_intro(self) -> None:
        self._settle()
        if self._stopped or not self.get_mapped() or not animations_enabled():
            return

        def pulse_tick(value: float) -> None:
            self.logo.pulse = value
            self.logo.queue_draw()

        pulse = Adw.TimedAnimation(
            widget=self.logo,
            value_from=0,
            value_to=1,
            duration=PULSE_MS,
            repeat_count=0,
            easing=Adw.Easing.LINEAR,
            target=Adw.CallbackAnimationTarget.new(pulse_tick),
        )
        self._animations.append(pulse)
        pulse.play()

    def _settle(self) -> None:
        if self.logo_settled:
            return
        self.logo_settled = True
        callbacks, self._settled_callbacks = self._settled_callbacks, []
        for callback in callbacks:
            callback()

    def when_settled(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once the logo's entrance is complete."""
        if self.logo_settled:
            callback()
        else:
            self._settled_callbacks.append(callback)

    def stop(self) -> None:
        """Stop every animation (the screen is no longer shown)."""
        self._stopped = True
        for animation in self._animations:
            animation.pause()

    # --- states --------------------------------------------------------------
    def set_status(self, text: str) -> None:
        self.status.set_label(text)

    def show_error(self, details: str) -> None:
        """Startup failed: say so plainly; the technical details stay folded away."""
        self.title.set_label("Unable to start Pixelift")
        self.title.remove_css_class("title-1")
        self.title.add_css_class("title-2")
        self.status.set_visible(False)
        self.dots.set_visible(False)
        self.details.set_label(details.strip())
        self.error_box.set_visible(True)
        for widget in (self.title, self.error_box):
            widget.set_opacity(1)
        self.retry_btn.grab_focus()

    def show_starting(self) -> None:
        """Back from an error to the starting state (Try Again)."""
        self.title.set_label(APP_NAME)
        self.title.remove_css_class("title-2")
        self.title.add_css_class("title-1")
        self.error_box.set_visible(False)
        self.status.set_visible(True)
        self.dots.set_visible(True)
        self.set_status(STATUS_STARTING)
