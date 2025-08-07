"""
Main entry point for the ID Photo Validator application.
"""

import tkinter as tk
from id_validator.gui import IDPhotoValidatorGUI

if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    
    # Instantiate the GUI class
    app = IDPhotoValidatorGUI(root)
    
    # Start the Tkinter event loop
    root.mainloop()
