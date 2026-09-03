"""
this file contains variables representing pre-configured pyqt stylesheets so they can be used in multiple places

they can be used by importing them and calling them like:
    label = QLabel("Hello with border!")
    label.setStyleSheet(style_sheets.red_label_border)

"""
from pathlib import Path
# path to assests
UI_DIR = Path(__file__).parent.parent
ICON_DIR = UI_DIR / "icons"

def update_stylesheet_property(widget, prop, value):
    """
    Updates or adds a single style property (e.g. 'border', 'padding') to the widget's stylesheet.

    Usage:
        update_stylesheet_property(my_widget, "border", "none")
        update_stylesheet_property(my_widget, "padding", "6px")
        update_stylesheet_property(my_widget, "background-color", "#f9f9f9")
    """
    style = widget.styleSheet().strip()
    lines = [line.strip() for line in style.split(";") if line.strip()]
    
    # Create a dict of existing styles
    style_dict = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            style_dict[k.strip()] = v.strip()

    # Update the desired property
    style_dict[prop] = value

    # Rebuild stylesheet
    new_style = "; ".join(f"{k}: {v}" for k, v in style_dict.items()) + ";"
    widget.setStyleSheet(new_style)


class StyleTemplate:
    def __init__(self, style_str: str):
        """ when called with a qt obj as argument returns style sheet string """
        self.style_str = style_str

    def __call__(self, widget_type: type) -> str:
        return self.style_str.format(widget=widget_type.__name__)
    
# error validation
##############################
red_border = StyleTemplate("""
{widget} {{
    border: 2px solid red;
    border-radius: 4px;
}}
""")

green_border = StyleTemplate("""
{widget} {{
    border: 2px solid green;
    border-radius: 4px;
}}
""")


def format_tooltip(text):
    """Format the tooltip using HTML and respect newlines via CSS."""
    
    formatted_text = text.replace('\n', '<br>')

    return (f"""
    <div style='width: 300px; background-color: #2b2b2b; color: white; padding: 8px; border-radius: 4px; border: 1px solid #444;'>
        <p style='font-size: 14px; font-weight: bold; margin-bottom: 4px; color: #b39ddb;'>
            Information
        </p>
        <p style='font-size: 13px; line-height: 1.4;'>
            {formatted_text}
        </p>
    </div>
    """)

# list widget
##############################
list_widget_style = """
    QListWidget {
        background-color: #1e1e1e;
        border: 2px solid #333;
        border-radius: 4px;
        padding: 5px;
        color: #ddd;
    }
    QListWidget::item {
        padding: 0.5px;
        border-bottom: 1px solid #2a2a2a;
        border-radius: 2px;
        margin-bottom: 4px;
    }
    QListWidget::item:selected {
        background-color: #30115c; /* Darker purple */
        color: white;
        font-weight: bold;
        border: 1px solid #794acf; /* Soft purple highlight border */
    }
    QListWidget::item:hover {
        background-color: #3b2e4d; /* Subtle dark purple to replace the dark blue hover */
    }
"""


# app tray tabs
##############################
app_tray_tabs = """
    /* 1. Target ONLY the top-level TabBar using '>' */
    QTabBar::tab {
        font-size: 12pt;
        font-weight: bold;
        padding: 4px 8px;
        min-width: 100px;
        
        /* THE 'COOL' LOOK */
        background: #3d3d3d;       /* Darker background for inactive tabs */
        color: #aaaaaa;            /* Dimmer text for inactive tabs */
        border: 1px solid #555;
        border-bottom: none;       /* Keep the bottom open to 'attach' to the pane */
        
        border-top-left-radius: 10px;  /* Rounded corners */
        border-top-right-radius: 10px;
        
        margin-right: 4px;         /* Gap between tabs */
        margin-top: 5px;           /* Pushes inactive tabs down slightly */
    }

    /* 2. Style the Selected Tab */
    QTabBar::tab:selected {
        background-color: white; 
        color: black;
        margin-top: 0px;           /* Makes the active tab look taller/closer */
        border-bottom: 2px solid white; /* Blends it into the content area */
    }

    /* 3. Style the hover state for interactivity */
    QTabBar::tab:hover:!selected {
        background-color: #4d4d4d;
        color: white;
    }

    /* 4. Fix the background pane (the box below the tabs) */
    QTabWidget::pane {
        border: 1px solid #555;
        top: -1px;                 /* Overlap the tab border for a seamless look */
        background-color: #2b2b2b; /* Prevents Windows Light Mode from making this white */
    }

    QTabWidget#AppTray > QTabBar::tab {
        font-size: 14pt;
        min-width: 150px;
    }
"""

# global dark theme
global_style = f"""
    /* Force main window and structural elements to dark */
    QMainWindow, QWidget {{
        background-color: #2b2b2b;
        color: #ffffff;
    }}
    
    /* Inner Tab Content Areas (Fixes the light gray background) */
    QTabWidget::pane {{
        background-color: #2b2b2b;
        border: 1px solid #555;
    }}

    /* Base Input Fields */
    QLineEdit, QComboBox, QPushButton {{
        background-color: #3d3d3d;
        color: #ffffff;
        border: 1px solid #555;
        padding: 4px;
        border-radius: 4px;
    }}
    
    /* Hover effects for inputs */
    QLineEdit:hover, QComboBox:hover, QPushButton:hover {{
        border: 1px solid #aaaaaa;
    }}

    /* =========================
       CHECKBOXES
       ========================= */
    QCheckBox {{
        spacing: 8px; 
        color: #ffffff;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        background-color: #3d3d3d;
        border: 1.5px solid #888888;
        border-radius: 3px;
    }}
    QCheckBox::indicator:hover {{
        border: 1.5px solid #ffffff; 
    }}
    QCheckBox::indicator:checked {{
        border: 1.5px solid #aaaaaa;
        background-color: #41167d; 
        image: url({ICON_DIR.as_posix()}/checkmark_white.png); 
    }}

    /* =========================
       SPIN BOXES
       ========================= */
    QSpinBox, QDoubleSpinBox {{
        background-color: #3d3d3d;
        color: #ffffff;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px; 
        padding-right: 25px; /* Prevent text hiding under buttons */
    }}
    QSpinBox:hover, QDoubleSpinBox:hover {{
        border: 1px solid #aaaaaa;
    }}

    /* Purple Buttons */
    QSpinBox::up-button, QDoubleSpinBox::up-button, 
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        width: 26px; 
        background-color: #26163d; 
        border-left: 1px solid #555; 
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: 3px; 
        border-bottom: 1px solid #555; 
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: 3px;
    }}
    
    /* Button Hover */
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: #9575cd; 
    }}

    /* Custom SVG Arrows */
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 20px;
        height: 20px;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({ICON_DIR.as_posix()}/chevron_up_white.svg); 
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({ICON_DIR.as_posix()}/chevron_down_white.svg); 
    }}
    QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off,
    QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {{
        opacity: 0.3;
    }}

    /* =========================
       COMBO BOXES (Dropdowns)
       ========================= */
    QComboBox {{
        padding-right: 30px; /* Protect text from the 26px button */
    }}
    
    QComboBox QAbstractItemView {{
        background-color: #3d3d3d;
        color: #ffffff;
        /* Sets the fallback highlight color */
        selection-background-color: #30115c; 
        border: 1px solid #555;
        /* Removes the dotted Windows focus border on clicked items */
        outline: none; 
    }}
    
    /* Explicitly style the dropdown items when hovered or selected */
    QComboBox QAbstractItemView::item:hover, 
    QComboBox QAbstractItemView::item:selected {{
        background-color: #30115c; 
        color: #ffffff;
    }}
    
    /* Purple Dropdown Button */
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 26px; 
        background-color: #26163d; 
        border-left: 1px solid #555; 
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }}
    QComboBox::drop-down:hover {{
        background-color: #9575cd; 
    }}
    
    /* Custom SVG Arrow */
    QComboBox::down-arrow {{
        image: url({ICON_DIR.as_posix()}/chevron_down_white.svg); 
        width: 20px;
        height: 20px;
    }}
    QComboBox::down-arrow:off, QComboBox::down-arrow:disabled {{
        opacity: 0.3;
    }}
    
    /* =========================
       TOOLTIPS
       ========================= */
    QToolTip {{
        background-color: #2b2b2b;
        color: #ffffff;
        border: 1px solid #7e57c2; /* Purple border to match the spin boxes and dropdowns */
        padding: 6px;
        border-radius: 4px;
    }}
    
"""