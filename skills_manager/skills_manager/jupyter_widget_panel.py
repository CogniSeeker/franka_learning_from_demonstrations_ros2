import ipywidgets as widgets
from IPython.display import display

class JupyterWidgetPanel():    
    def widget(self):
        # --- Output area where results will appear ---
        self.widget_out = widgets.Output()

        btn1 = widgets.Button(
            description="(ON) Franka Buttons" if self.frankabuttons_running else "(OFF) Franka Buttons",
            button_style="success" if self.frankabuttons_running else "",
            tooltip="frankabuttons_stop(), frankabuttons_start()",
        )

        btn2 = widgets.Button(
            description="(ON) Joystick" if self.joy_thread_running else "(OFF) Joystick",
            button_style="success" if self.joy_thread_running else "",
            tooltip="joy_stop(), joy_start()",
        )

        btn3 = widgets.Button(
            description="(ON) Hand Teleoperation" if self.teleop_thr_running else "(OFF) Hand Teleoperation",
            button_style="success" if self.teleop_thr_running else "",
            tooltip="teleop_stop(), teleop_start()",
        )

        btn4 = widgets.Button(
            description="(ON) Keyboard" if self.key_thr_running else "(OFF) Keyboard",
            button_style="success" if self.key_thr_running else "",
            tooltip="keyboard_stop(), keyboard_start()",
        )

        btn5 = widgets.Button(
            description= "Grasped" if (self._robot is not None and self._robot.gripper_state.is_grasped) else "Open",
            button_style="success" if (self._robot is not None and self._robot.gripper_state.is_grasped) else "",
            tooltip="keyboard_stop(), keyboard_start()",
        )

        btn6 = widgets.Button(
            description= "Lock",
            button_style="warning",
            tooltip="desk.lock(), desk.release_control()",
        )

        btn7 = widgets.Button(
            description= "Unlock",
            button_style="info",
            tooltip="desk.take_control(), desk.unlock(), desk.activate_fci()",
        )

        btn8 = widgets.Button(
            description= "Homing",
            button_style="",
            tooltip="home_gripper(), home()",
        )

        btn1.on_click(self.widget_btn1_clicked)
        btn2.on_click(self.widget_btn2_clicked)
        btn3.on_click(self.widget_btn3_clicked)
        btn4.on_click(self.widget_btn4_clicked)
        btn5.on_click(self.widget_btn5_clicked)
        btn6.on_click(self.widget_btn6_clicked)
        btn7.on_click(self.widget_btn7_clicked)
        btn8.on_click(self.widget_btn8_clicked)

        # --- Layout "dashboard panel" ---
        group1_label = widgets.HTML("<b>Enable Teach Input</b>")
        group1_row = widgets.HBox([group1_label, btn1, btn2, btn3, btn4])

        group1 = widgets.VBox([group1_row, self.widget_out])

        # --- Group 2: Pipeline controls ---
        group2_label = widgets.HTML(f"<b>Robot</b> {'On Real-time kernel!' if self._robot.has_realtime_kernel() else 'Not on real-time kernel, controller might restart occasionally'}")

        group2_label2 = widgets.HTML("<b> | </b> Gripper: ")
        group2_row = widgets.HBox([btn6, btn7, group2_label2, btn5, btn8])

        group2 = widgets.VBox([group2_label, group2_row])

        # --- Whole dashboard panel ---
        return widgets.VBox([
            group1,
            widgets.HTML("<hr>"),
            group2,
        ])

    # --- Some functions you want to call ---
    def widget_toggle_frankabuttons(self, b):
        if self.frankabuttons_running:
            self.frankabuttons_stop() # Not implemented
            # b.description = "(OFF) Franka Buttons"
            # b.button_style = ""
            # print("Franka buttons stopped...")
        else:
            self.frankabuttons_start()
            b.description = "(ON) Franka Buttons"
            b.button_style = "success"
            print("Franka buttons started...")

    def widget_toggle_joy(self, b):
        if self.joy_thread_running:
            self.joy_stop()
            b.description = "(OFF) Joystick"
            b.button_style = ""
            print("Joystick listener stopped...")
        else:
            self.joy_start()
            b.description = "(ON) Joystick"
            b.button_style = "success"
            print("Joystick listener started...")

    def widget_toggle_teleop(self, b):
        if self.teleop_thr_running:
            self.teleop_stop()
            b.description = "(OFF) Hand Teleoperation"
            b.button_style = ""
            print("Hand teleoperation listener stopped...")
        else:
            self.teleop_start()
            b.description = "(ON) Hand Teleoperation"
            b.button_style = "success"
            print("Hand teleoperation listener started...")

    def widget_toggle_keyboard(self, b):
        if self.key_thr_running:
            self.keyboard_stop()
            b.description = "(OFF) Keyboard"
            b.button_style = ""
            print("Keyboard listener stopped...")
        else:
            self.keyboard_start()
            b.description = "(ON) Keyboard"
            b.button_style = "success"
            print("Keyboard listener started...")

    def widget_toggle_gripper(self, b):
        if self._robot is None:
            print("_robot not defined")
            return

        if not self._robot.gripper_state.is_grasped:
            print(f"is_grasped is False -> Grasping")
            self._robot.grasp_gripper(0)
            b.description = "Grasped"
            b.button_style = "success"
            print(f"Grasped")
        else:
            print(f"is_grasped is True -> Opening")
            self._robot.move_gripper(0.08)
            b.description = "Open"
            b.button_style = ""
            print(f"Opened")

    def widget_lock_robot(self, b):    
        self.desk.lock()
        self.desk.release_control()

    def widget_unlock_robot(self, b):    
        self.desk.take_control()
        self.desk.unlock()
        self.desk.activate_fci()

    def widget_home(self, b):
        self.home_gripper()
        self.home()

    # --- Button callbacks ---
    def widget_btn1_clicked(self, b):
        with self.widget_out:  # send prints/outputs into this panel
            self.widget_out.clear_output()
            self.widget_toggle_frankabuttons(b)

    def widget_btn2_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_toggle_joy(b)

    def widget_btn3_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_toggle_teleop(b)

    def widget_btn4_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_toggle_keyboard(b)

    def widget_btn5_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_toggle_gripper(b)

    def widget_btn6_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_lock_robot(b)

    def widget_btn7_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_unlock_robot(b)

    def widget_btn8_clicked(self, b):
        with self.widget_out:
            self.widget_out.clear_output()
            self.widget_home(b)
