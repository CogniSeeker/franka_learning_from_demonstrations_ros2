import tkinter as tk


class Signalizator:
    # You can tune these colors as you like
    COLOR_IDLE = "#7f8c8d"          # gray
    COLOR_DEMONSTRATION = "#f1c40f" # yellow
    COLOR_DEMONSTRATION_READY = "#bb9b1d" # yellow
    COLOR_EXECUTION = "#2ecc71"     # green

    def __init__(self, fullscreen=True, monitor_offset_x=0, size="400x200"):
        self.root = tk.Tk()
        self.root.title("Robot status")

        if fullscreen:
            # fill one monitor, shifted by monitor_offset_x pixels (e.g. 1920)
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.geometry(f"800x600-3500+0")
        else:
            # normal, resizable window
            self.root.geometry(f"400x200-1920+0")
            self.root.resizable(True, True)

        self.label = tk.Label(self.root, font=("DejaVu Sans", 150, "bold"))
        self.label.pack(expand=True, fill="both")

        self.signalize_idle()


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