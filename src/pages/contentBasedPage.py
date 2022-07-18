import tkinter as tk
import tkinter.messagebox
import customtkinter
from pandastable import Table, TableModel
from recommenders.contentBased import *

class ContentBasedPage(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__()
        
        customtkinter.CTkFrame.__init__(self)
        
        # Create recommender and get category list
        self.contentRec = ContentBasedRecommender()

        self.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)
        self.rowconfigure(7, weight=2)
        self.columnconfigure((0, 2,), weight=1)
        self.columnconfigure((1, 3), weight=0)

        # Default values
        self.movie_title = "The Dark Knight"
        self.filter = 'Description'
        self.showed_results = 10

        # Page name
        self.page_name = customtkinter.CTkLabel(master=self,
                                            text="Content Based Recommender",
                                            text_font=("Roboto Medium", 28))
        self.page_name.grid(row=0, column=0, columnspan=4, sticky="s")

        # Title Input
        self.title_label = customtkinter.CTkLabel(master=self, text="Type the movie title below:")

        self.title_label.grid(row=1, column=0, pady=0, padx=20, sticky="ws")

        self.title_input = customtkinter.CTkEntry(master=self,
                                            placeholder_text=self.movie_title)
        self.title_input.grid(row=2, column=0, pady=0, padx=20, sticky="wen")

        # Search button
        self.search_btn = customtkinter.CTkButton(master=self,
                                                text="Search",
                                                command= self.update_title)
        self.search_btn.grid(row=2, column=1, pady=0, padx=20, sticky='wen')

        # Filter       
        self.filter_label = customtkinter.CTkLabel(master=self, text="Choose filter by:")

        self.filter_label.grid(row=1, column=2, pady=0, padx=20, sticky="ws")

        self.filter_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=["Description", "Metadata", "Metadata and Rating"],
                                                        command= self.update_filter)

        self.filter_option.grid(row=2, column=2, pady=0, padx=20, sticky='ewn')

        # Showed results
        self.showed_results_label = customtkinter.CTkLabel(master=self, text="Choose number of result to show:")

        self.showed_results_label.grid(row=1, column=3, pady=0, padx=20, sticky="ws")

        self.showed_results_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=["10", "25", "50", "100"],
                                                        command= self.update_showed_result)

        self.showed_results_option.grid(row=2, column=3, pady=0, padx=20, sticky='ewn')

        # Pandas table
        self.dataframe = customtkinter.CTkFrame(master=self,                                                
                                                 corner_radius=0)
        self.dataframe.grid(row=3, column=0, columnspan=4, rowspan=4, padx=20, sticky="nswe")

        df = self.contentRec.recommendByDescription(self.movie_title).head(self.showed_results)    
        df.columns = ['Title', 'Description']    
        self.table = Table(self.dataframe, dataframe=df,
                                showtoolbar=True, showstatusbar=True, editable=False)
        
        self.table.show()
        self.table.zoomIn()
        self.table.autoResizeColumns()
    
    def update_title(self):
        self.movie_title = self.title_input.get()
        self.update_filter(self.filter)

    def update_filter(self, filter):
        self.filter = filter

        if (filter == "Description"):
            self.table.model.df = self.contentRec.recommendByDescription(self.movie_title).head(self.showed_results)
            self.table.model.df.columns = ['Title', 'Description']     

        if (filter == "Metadata"):
            self.table.model.df = self.contentRec.recommendByMetadata(self.movie_title).head(self.showed_results)
            self.table.model.df.columns = ['Title', 'Description']    

        if (filter == "Metadata and Rating"):
            self.table.model.df = self.contentRec.recommendByMetadataAndRating(self.movie_title).head(self.showed_results)
            self.table.model.df.columns = ['Id', 'Title', 'Release date', 'Categories', 'Vote count',
                    'Vote average', 'Popularity', 'Weighted rating']     

        self.table.redraw()

    def update_showed_result(self, result):
        self.showed_results = int(result)
    
        self.update_filter(self.filter)