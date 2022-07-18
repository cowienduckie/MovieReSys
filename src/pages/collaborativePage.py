import tkinter as tk
import tkinter.messagebox
import customtkinter
from pandastable import Table, TableModel
from recommenders.collaborativeBased import *

class CollaborativeBasedPage(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__()
        
        customtkinter.CTkFrame.__init__(self)
        
        # Create recommender
        self.collaborativeRec = CollaborativeBasedRecommender()
        self.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # Config rows and cols
        self.rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)
        # self.rowconfigure(7, weight=2)
        self.columnconfigure((1, 2), weight=0)
        self.columnconfigure(0, weight=1)

        # Default values
        self.userid = 1
        self.showed_results = 10

        # Page name
        self.page_name = customtkinter.CTkLabel(master=self,
                                            text="Collaborative Based Recommender",
                                            text_font=("Roboto Medium", 28))
        self.page_name.grid(row=0, column=0, columnspan=3, sticky="s")

        # User ID Input
        self.userid_label = customtkinter.CTkLabel(master=self, text="Type the User Id below:")

        self.userid_label.grid(row=1, column=0, pady=0, padx=20, sticky="ws")

        self.userid_input = customtkinter.CTkEntry(master=self,
                                            placeholder_text=self.userid)
        self.userid_input.grid(row=2, column=0, pady=0, padx=20, sticky="wen")

        # Search button
        self.search_btn = customtkinter.CTkButton(master=self,
                                                text="Search",
                                                command= self.update_userid)
        self.search_btn.grid(row=2, column=1, pady=0, padx=20, sticky='wen')

        # Showed results
        self.showed_results_label = customtkinter.CTkLabel(master=self, text="Choose number of result to show:")

        self.showed_results_label.grid(row=1, column=2, pady=0, padx=20, sticky="ws")

        self.showed_results_option = customtkinter.CTkOptionMenu(master=self,
                                                        values=["10", "25", "50", "100"],
                                                        command= self.update_showed_result)

        self.showed_results_option.grid(row=2, column=2, pady=0, padx=20, sticky='ewn')

        # User ratings table
        self.user_ratings_frame = customtkinter.CTkFrame(master=self,                                                
                                                 corner_radius=0)
        self.user_ratings_frame.grid(row=3, column=0, columnspan=3, rowspan=1, padx=20, sticky="nswe")

        ratings = self.collaborativeRec.getUserRatings(self.userid).head(10)
        ratings.columns = ['Title', 'Description', 'Rating']
        self.user_ratings_table = Table(self.user_ratings_frame, dataframe=ratings,
                                showtoolbar=False, showstatusbar=False, editable=False)
        
        self.user_ratings_table.show()
        self.user_ratings_table.zoomIn()
        self.user_ratings_table.autoResizeColumns()

        # Pandas table
        self.dataframe = customtkinter.CTkFrame(master=self,                                                
                                                 corner_radius=0)
        self.dataframe.grid(row=4, column=0, columnspan=3, rowspan=3, padx=20, sticky="nswe")

        df = self.collaborativeRec.recommend(self.userid).head(self.showed_results)    
        df.columns = ['Title', 'Description', 'Estimated Rating']    
        self.table = Table(self.dataframe, dataframe=df,
                                showtoolbar=False, showstatusbar=True, editable=False)
        
        self.table.show()
        self.table.zoomIn()
        self.table.autoResizeColumns()
    
    def update_userid(self):
        self.userid = int(self.userid_input.get())

        # Update recommends
        self.table.model.df = self.collaborativeRec.recommend(self.userid).head(self.showed_results)
        self.table.model.df.columns = ['Title', 'Description', 'Estimated Rating']
        self.table.redraw()

        # Update ratings
        self.user_ratings_table.model.df = self.collaborativeRec.getUserRatings(self.userid).head(10)
        self.user_ratings_table.model.df.columns = ['Title', 'Description', 'Rating']
        self.user_ratings_table.redraw()

    def update_showed_result(self, result):
        self.showed_results = int(result)
    
        # Update recommends
        self.table.model.df = self.collaborativeRec.recommend(self.userid).head(self.showed_results)
        self.table.model.df.columns = ['Title', 'Description', 'Estimated Rating']
        self.table.redraw()

        # Update ratings
        self.user_ratings_table.model.df = self.collaborativeRec.getUserRatings(self.userid).head(10)
        self.user_ratings_table.model.df.columns = ['Title', 'Description', 'Rating']
        self.user_ratings_table.redraw()