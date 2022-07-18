import tkinter as tk
import tkinter.messagebox
import customtkinter
from pandastable import Table, TableModel
from recommenders.simple import *

class SimplePage(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__()
        
        customtkinter.CTkFrame.__init__(self)
        self.simpleRec = SimpleRecommender()

        self.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.rowconfigure((0, 1, 2, 3), weight=1)
        self.rowconfigure(7, weight=10)
        self.columnconfigure((0, 1), weight=1)
        self.columnconfigure(2, weight=0)

        # Default values
        self.category = 'Romance'
        self.showed_results = 10

        # Components
        self.label = customtkinter.CTkLabel(master=self,
                                            text="Simple Recommender",
                                            text_font=("Roboto Medium", 28))
        self.label.grid(row=0, column=0, columnspan=3 )

        category_list = self.simpleRec.get_category_list

        self.category_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=category_list,
                                                        command= self.update_category)

        self.category_option.grid(row=1, column=1, pady=10, padx=20, sticky='ew')
        # self.category_option.config(width = len(self.category_option))

        self.showed_results_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=["10", "25", "50", "100"],
                                                        command= self.update_showed_result)

        self.showed_results_option.grid(row=1, column=2, pady=10, padx=20, sticky='ew')

        # Pandas table
        self.dataframe = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.dataframe.grid(row=2, column=0, columnspan=3, padx=20, sticky="nswe")

        df = self.simpleRec.build_chart(self.category).head(self.showed_results)        
        self.table = Table(self.dataframe, dataframe=df,
                                showtoolbar=True, showstatusbar=True)
        self.table.show()

    def update_category(self, category):
        self.category = category

        self.table.model.df = self.simpleRec.build_chart(self.category).head(self.showed_results)   

        self.table.redraw()

    def update_showed_result(self, result):
        self.showed_results = int(result)
        
        self.table.model.df = self.simpleRec.build_chart(self.category).head(self.showed_results)   

        self.table.redraw()