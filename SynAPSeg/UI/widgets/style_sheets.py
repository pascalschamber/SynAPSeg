"""
this file contains variables representing pre-configured pyqt stylesheets so they can be used in multiple places

they can be used by importing them and calling them like:
    label = QLabel("Hello with border!")
    label.setStyleSheet(style_sheets.red_label_border)

"""

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

# tooltip
##############################
tooltip_style ="""
    QToolTip {
        background-color: #2b2b2b;
        color: white;
        border: 1px solid #444;
        padding: 5px;
        opacity: 230; /* Slight transparency */
    }
"""

def format_tooltip(text):
    """Format the tooltip using HTML and respect newlines via CSS."""
    
    formatted_text = text.replace('\n', '<br>')

    return (f"""
    <div style='width: 300px; background-color: #2b2b2b; color: white; padding: 8px; border-radius: 4px; border: 1px solid #444;'>
        <p style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #3498db;'>
            Information
        </p>
        <p style='font-size: 12px; line-height: 1.4;'>
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
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: 1px solid #5dade2;
    }
    QListWidget::item:hover {
        background-color: #2c3e50;
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
    QTabBar::pane {
        border: 1px solid #555;
        top: -1px;                 /* Overlap the tab border for a seamless look */
    }

    QTabWidget#AppTray > QTabBar::tab {
        font-size: 14pt;
        min-width: 150px;
    }
"""