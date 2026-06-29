import os
import tkinter as tk

# Remember where the user last placed/sized the window between runs.
GEOMETRY_FILE = os.path.expanduser("~/.config/robot_signalizer/geometry.txt")


class Signalizator:
    # You can tune these colors as you like
    COLOR_IDLE = "#7f8c8d"          # gray
    COLOR_DEMONSTRATION = "#f1c40f" # yellow
    COLOR_DEMONSTRATION_READY = "#bb9b1d" # yellow
    COLOR_EXECUTION = "#2ecc71"     # green

    def __init__(self, fullscreen=True, monitor_offset_x=0, size="400x200"):
        self.root = tk.Tk()
        self.root.title("Robot status")

        saved_geometry = self._load_geometry()
        if saved_geometry:
            # restore last position/size from the previous run
            self.root.geometry(saved_geometry)
            self.root.resizable(True, True)
        elif fullscreen:
            # fill one monitor, shifted by monitor_offset_x pixels (e.g. 1920)
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.geometry(f"1280x640-4000+0")
        else:
            # normal, resizable window
            self.root.geometry(f"400x200-1920+0")
            self.root.resizable(True, True)

        # Persist the geometry whenever the window is moved or resized.
        self._last_saved_geometry = saved_geometry
        self.root.bind("<Configure>", self._on_configure)

        self.label = tk.Label(self.root, font=("DejaVu Sans", 150, "bold"))
        self.label.pack(expand=True, fill="both")

        # There is no blocking mainloop (Signalizator.run() is not called when
        # driven from lfd), so the kernel would never pump tkinter events while
        # idle: the window could be dragged by the WM but the app would never see
        # the move, so the geometry would never get saved. enable_gui("tk")
        # integrates the tk event loop into the IPython prompt (like %gui tk).
        self._enable_ipython_tk()

        self.signalize_idle()

    def _enable_ipython_tk(self):
        try:
            from IPython.core.getipython import get_ipython
            ip = get_ipython()
            if ip is not None:
                ip.enable_gui("tk")
        except Exception:
            pass

    def _load_geometry(self):
        """Return the saved 'WxH+X+Y' geometry string, or None if unavailable."""
        try:
            with open(GEOMETRY_FILE) as f:
                geometry = f.read().strip()
            return geometry or None
        except OSError:
            return None

    def _on_configure(self, event):
        # Ignore Configure events bubbling up from child widgets (e.g. the label).
        if event.widget is not self.root:
            return
        self._save_geometry()

    def _save_geometry(self):
        geometry = self.root.geometry()  # 'WxH+X+Y'
        # Skip the bogus "1x1+0+0" tkinter reports before the window is mapped,
        # and avoid redundant writes when nothing moved.
        if geometry.startswith("1x1") or geometry == self._last_saved_geometry:
            return
        try:
            os.makedirs(os.path.dirname(GEOMETRY_FILE), exist_ok=True)
            with open(GEOMETRY_FILE, "w") as f:
                f.write(geometry)
            self._last_saved_geometry = geometry
        except OSError:
            pass

    def _set_state(self, text: str, color: str):
        """Internal helper to change background + text."""
        self.root.configure(bg=color)
        self.label.configure(text=text, bg=color, fg="black")
        # Ensure redraw happens immediately if called while running
        self.root.update_idletasks()

    def signalize_demonstration(self):
        self._set_state("IS\nDEMONSTRATING", self.COLOR_DEMONSTRATION)

    def signalize_ready_demonstration(self):
        self._set_state("IS\nREADY!", self.COLOR_DEMONSTRATION_READY)

    def signalize_execution(self):
        self._set_state("IS\nEXECUTING", self.COLOR_EXECUTION)

    def signalize_idle(self):
        self._set_state("IS\nIDLE", self.COLOR_IDLE)

    def run(self):
        """Blocking call – starts the event loop."""
        self.root.mainloop()

    def close(self):
        if self.root is not None and self.root.winfo_exists():
            self.root.after(0, self.root.destroy)


if __name__ == "__main__":
    s = Signalizator()
    s.run()