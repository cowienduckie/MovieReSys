import tkinter as tk
import tkinter.messagebox
import customtkinter
from pandastable import Table, TableModel
from recommenders.simple import *

class SimplePage(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__()
        
        customtkinter.CTkFrame.__init__(self)
        
        # Create recommender and get category list
        self.simpleRec = SimpleRecommender()
        category_list = self.simpleRec.get_category_list()
        category_list.sort()

        self.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)
        self.rowconfigure(7, weight=2)
        self.columnconfigure((0, 1, 2), weight=1)

        # Default values
        self.category = category_list[0]
        self.showed_results = 10

        # Page name
        self.page_name = customtkinter.CTkLabel(master=self,
                                            text="Simple Recommender",
                                            text_font=("Roboto Medium", 28))
        self.page_name.grid(row=0, column=0, columnspan=3, sticky="s")

        # Category filter       
        self.category_label = customtkinter.CTkLabel(master=self, text="Choose a category:")

        self.category_label.grid(row=1, column=0, pady=0, padx=0, sticky="ws")

        self.category_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=category_list,
                                                        command= self.update_category)

        self.category_option.grid(row=2, column=0, pady=0, padx=20, sticky='ewn')

        # Showed results
        self.showed_results_label = customtkinter.CTkLabel(master=self, text="Choose number of result to show:")

        self.showed_results_label.grid(row=1, column=1, pady=0, padx=20, sticky="ws")

        self.showed_results_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=["10", "25", "50", "100"],
                                                        command= self.update_showed_result)

        self.showed_results_option.grid(row=2, column=1, pady=0, padx=20, sticky='ewn')

        # Pandas table
        self.dataframe = customtkinter.CTkFrame(master=self,                                                
                                                 corner_radius=0)
        self.dataframe.grid(row=3, column=0, columnspan=3, rowspan=4, padx=20, sticky="nswe")

        df = self.simpleRec.build_chart(self.category).head(self.showed_results)    
        df.columns = ['Id', 'Title', 'Release date', 'Categories', 'Vote count',
                    'Vote average', 'Popularity', 'Weighted rating']    
        self.table = Table(self.dataframe, dataframe=df,
                                showtoolbar=True, showstatusbar=True, editable=False)
        
        self.table.show()
        self.table.zoomIn()
        self.table.autoResizeColumns()

    def update_category(self, category):
        self.category = category

        self.table.model.df = self.simpleRec.build_chart(self.category).head(self.showed_results)   

        self.table.redraw()

    def update_showed_result(self, result):
        self.showed_results = int(result)
        
        self.table.model.df = self.simpleRec.build_chart(self.category).head(self.showed_results)   

        self.table.redraw()