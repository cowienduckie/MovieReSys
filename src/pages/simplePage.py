import tkinter as tk
import tkinter.messagebox
import customtkinter
from pandastable import Table, TableModel
from recommenders.simple import *

class SimplePage(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__()
        
        customtkinter.CTkFrame.__init__(self)

        self.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.rowconfigure((0, 1, 2, 3), weight=1)
        self.rowconfigure(7, weight=10)
        self.columnconfigure((0, 1), weight=1)
        self.columnconfigure(2, weight=0)

        self.label = customtkinter.CTkLabel(master=self,
                                            text="Simple Recommender",
                                            text_font=("Roboto Medium", 10))
        self.label.grid(row=0, column=0, columnspan=1 )

        self.input = customtkinter.CTkEntry(master=self,
                                            width=120,
                                            placeholder_text="Input here")
        self.input.grid(row=1, column=0, columnspan=2, sticky="we")

        self.searchBtn = customtkinter.CTkButton(master=self,
                                                text="Search",
                                                border_width=2,
                                                fg_color=None,
                                                command=self.button_event)
        self.searchBtn.grid(row=1, column=2, columnspan=1, sticky="we")

        self.dataframe = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.dataframe.grid(row=2, column=0, sticky="nswe")

        simpleRec = SimpleRecommender()
        df = simpleRec.build_chart('Romance').head(10)
        
        self.table = pt = Table(self.dataframe, dataframe=df,
                                    showtoolbar=True, showstatusbar=True)
        pt.show()



    def button_event(self):
        print("Button pressed")