import tkinter as tk
import tkinter.messagebox
import customtkinter

class ContentBasedPage(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__()
        
        customtkinter.CTkFrame.__init__(self)

        self.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.rowconfigure((0, 1, 2, 3), weight=1)
        self.rowconfigure(7, weight=10)
        self.columnconfigure((0, 1), weight=1)
        self.columnconfigure(2, weight=0)

        self.label = customtkinter.CTkLabel(master=self,
                                            text="Content Based Recommender",
                                            text_font=("Roboto Medium", 10))
        self.label.grid(row=0, column=0, columnspan=1 )

        self.entry = customtkinter.CTkEntry(master=self,
                                            width=120,
                                            placeholder_text="Input here")
        self.entry.grid(row=1, column=0, columnspan=2, sticky="we")

        self.button_5 = customtkinter.CTkButton(master=self,
                                                text="Search",
                                                border_width=2,
                                                fg_color=None,
                                                command=self.button_event)
        self.button_5.grid(row=1, column=2, columnspan=1, sticky="we")

    def button_event(self):
        print("Button pressed")