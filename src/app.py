import tkinter as tk
import tkinter.messagebox
import customtkinter
from pages.simplePage import *
from pages.contentBasedPage import *
from pages.collaborativePage import *
from pages.hybridPage import *

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()

        self.title("Movie ReSys")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        # Add page
        self.frames = {}
        self.frames["ContentBasedPage"] = ContentBasedPage(parent=self.frame_right)
        self.frames["ContentBasedPage"].grid(row=0, column=1, sticky="nsew")
        self.frames["CollaborativeBasedPage"] = CollaborativeBasedPage(parent=self.frame_right)
        self.frames["CollaborativeBasedPage"].grid(row=0, column=1, sticky="nsew")
        self.frames["HybridPage"] = HybridPage(parent=self.frame_right)
        self.frames["HybridPage"].grid(row=0, column=1, sticky="nsew")
        self.frames["SimplePage"] = SimplePage(parent=self.frame_right)
        self.frames["SimplePage"].grid(row=0, column=1, sticky="nsew")

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(2, minsize=10)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(7, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.app_name = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Movie Recommender System",
                                              text_font=("Roboto Medium", 16))  # font name and size in px
        self.app_name.grid(row=1, column=0, pady=10, padx=10)

        self.simple_btn = customtkinter.CTkButton(master=self.frame_left,
                                                text="Simple Recommender",
                                                command=lambda: self.show_frame("SimplePage"))
        self.simple_btn.grid(row=3, column=0, pady=10, padx=20, sticky='nesw')
    
        self.content_btn = customtkinter.CTkButton(master=self.frame_left,
                                                text="Content Based Recommender",
                                                command=lambda: self.show_frame("ContentBasedPage"))
        self.content_btn.grid(row=4, column=0, pady=10, padx=20, sticky='nesw')

        self.collab_btn = customtkinter.CTkButton(master=self.frame_left,
                                                text="Collaborative Based Recommender",
                                                command=lambda: self.show_frame("CollaborativeBasedPage"))
        self.collab_btn.grid(row=5, column=0, pady=10, padx=20, sticky='nesw')

        self.hybrid_btn = customtkinter.CTkButton(master=self.frame_left,
                                                text="Hybrid Recommender",
                                                command=lambda: self.show_frame("HybridPage"))
        self.hybrid_btn.grid(row=6, column=0, pady=10, padx=20, sticky='nesw')

        # Change theme
        self.theme_label = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.theme_label.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.theme_option = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.theme_option.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
        self.state('zoomed')

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.state('zoomed')
    app.mainloop()