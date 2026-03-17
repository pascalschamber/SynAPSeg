from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QWidget,
    QComboBox,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal

import traceback
import uuid
from pprint import pformat

from SynAPSeg.UI.widgets.container_widgets import ScrollableContainerWidget, CollapsibleWidget
from SynAPSeg.UI.widgets.config_fields import field_widget
from SynAPSeg.UI.widgets.dialogs import warning_dialog
from SynAPSeg.UI.widgets.label import HoverLabel
from SynAPSeg.config.param_engine.interpreter import SchemaInterpreter
from SynAPSeg.utils import utils_general as ug

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def format_widget_value(value):
    """
    Apply standard formatting to result of widget.get_value().
    Example: convert 'None' string to actual None type.
    """
    if isinstance(value, str):
        if value.lower() == 'none' or len(value) == 0:
            return None
    return value

def create_widget_from_spec(spec, on_change_callback=None):
    """
    Factory function to create a UI widget from a configuration specification.
    
    Args:
        spec (dict): The parameter specification (must contain 'heading', 'category', 'widget_type', etc.)
        on_change_callback (callable, optional): Function to call when value changes.
    """
    widget_kwargs = spec.get('widget_kwargs', {}).copy()
    
    # Handle category logic (group vs category)
    # group attr from config sometimes supersedes category
    category = spec.get('category')
    grp = spec.get('group')
    if grp is not None:
        category = None if grp == 'root' else grp
        
    attributes = {
        'default_value': spec.get('current_value', None),
        'widget_type': spec.get('widget_type', None),
        'tooltip': spec.get('tooltip', ''),
        'heading': spec.get('heading'),
        'category': category,
        'flags': spec.get('flags', None),
        **widget_kwargs
    }
    
    widget = field_widget(attributes)
    
    if on_change_callback and hasattr(widget, 'value_changed'):
        widget.value_changed.connect(on_change_callback)
        
    return widget


# -----------------------------------------------------------------------------
# Component Widgets
# -----------------------------------------------------------------------------


class StaticParamsGroup(QWidget):
    """
    Represents a group of static parameters (like 'Run Configuration') displayed in a FormLayout.
    """
    def __init__(self, group_name, specs, registry, on_change_callback):
        """
        Args:
            group_name (str): Title of the group (tab name).
            specs (dict): Dictionary of {scope: spec} for this group.
            registry (dict): Reference to master widget registry to register created widgets.
            on_change_callback (callable): Signal handler for value changes.
        """
        super().__init__()
        self.group_name = group_name
        self.layout = QFormLayout()
        self.layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)
        
        # Sort specs? Usually they come ordered or dict preserves insertion order in modern Py.
        for scope, spec in specs.items():
            widget = create_widget_from_spec(spec, on_change_callback)
            
            # Register widget
            registry[scope] = widget
            
            # Add to layout if visual
            if hasattr(widget, 'get_widget'):
                self.layout.addRow(HoverLabel(spec['name'], spec.get('tooltip')), widget.get_widget())


class AddModelWidget(QWidget):
    """
    Widget to handle adding new plugin components (Models) to the pipeline.
    """
    def __init__(self, options, add_callback):
        super().__init__()
        self.add_callback = add_callback
        
        self.combo = QComboBox()
        self.combo.addItems(options)
        
        self.model_name_field = QLineEdit()
        self.model_name_field.setPlaceholderText("Enter unique model name")
        
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self._on_add)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel('Select type:'))
        layout.addWidget(self.combo)
        layout.addWidget(self.model_name_field)
        layout.addWidget(self.add_button)
        self.setLayout(layout)

    def _on_add(self):
        model_class = self.combo.currentText()
        model_name = self.model_name_field.text().strip()
        if not model_name:
            model_name = model_class
            
        self.add_callback(model_class, model_name)


class PluginManagerWidget(QWidget):
    """
    Manages the dynamic list of plugins (e.g. Models).
    Contains a ScrollableContainerWidget for the list and an AddModelWidget.
    """
    def __init__(self, 
                 plugin_heading, 
                 plugin_param_key, 
                 plugin_class_key, 
                 plugin_factory, 
                 available_models,
                 registry, 
                 on_change_callback,
                 debug=False):
        super().__init__()
        self.plugin_heading = plugin_heading
        self.plugin_param_key = plugin_param_key # e.g. MODEL_PARAMS
        self.plugin_class_key = plugin_class_key
        self.plugin_factory = plugin_factory
        self.available_models = available_models
        self.registry = registry
        self.on_change_callback = on_change_callback
        self.debug = debug
        
        self.plugin_specs = {} # Stores specs for active plugins {scope: spec}
        self.active_models_widgets = {} # {model_name: {scope: widget}}
        self.collapsible_container = None
        
        # Layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # 1. Scrollable Container for Models
        # We start with empty dict, will populate via update_from_specs
        self.collapsible_container = ScrollableContainerWidget(
            {}, 
            collapsable_widget_kwargs={'delete_callback': self.delete_model}
        )
        self.main_layout.addWidget(self.collapsible_container)
        
        # 2. Add Model Widget
        self.add_widget = AddModelWidget(
            options=list(self.available_models.keys()), 
            add_callback=self.add_model
        )
        # Add 'Add Widget' to the bottom of the scroll content area
        self.collapsible_container.scroll_content.layout().addWidget(self.add_widget)
        
        # Add wrapping collapsible for the adder itself
        # (This matches original UI: "add a component" is a collapsible)
        container_for_adder = QWidget()
        l = QVBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.addWidget(self.add_widget)
        container_for_adder.setLayout(l)
        
        # Note: The original code wrapped AddModelWidget in a CollapsibleWidget named "add a component"
        # We can replicate strictly or simplify. Let's replicate strictly to maintain UI feel.
        self.adder_collapsible = CollapsibleWidget('add a component', container_for_adder, deletable=False)
        self.collapsible_container.scroll_content.layout().addWidget(self.adder_collapsible)


    def update_from_specs(self, full_plugin_specs):
        """
        Rebuilds the UI based on full scoped specifications for the plugin section.
        args:
            full_plugin_specs (dict): e.g. {'Model.MODEL_PARAMS.current_value...': spec}
        """
        # Clear existing models in UI
        # self.collapsible_container.clear_widgets() # Helper method we might need to assume or use logic below
        # ScrollableContainerWidget doesn't seem to have a clear_widgets method in previous usage, 
        # but in original code: _clear_model_param_widgets() cleared internal dicts. 
        # To clear UI, we must remove from layout.
        
        # Logic to clear the scroll_content layout except the adder
        layout = self.collapsible_container.scroll_content.layout()
        while layout.count() > 1: # Keep the adder (which is last)
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
            
        # Re-init state
        self.plugin_specs = full_plugin_specs
        self.active_models_widgets = {}
        
        # Identify models
        model_names = self._get_model_names_from_specs(full_plugin_specs)
        
        for model_name in model_names:
            # Extract specs for this model
            model_specs = {k:v for k,v in full_plugin_specs.items() 
                           if self._get_model_name_from_scope(k) == model_name}
            
            # Build UI for this model
            self._build_model_ui(model_name, model_specs)


    def _build_model_ui(self, model_name, model_specs):
        """Builds valid QFormLayout for a single model and adds to container."""
        param_form = QFormLayout()
        param_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        param_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        headings_added = set()
        
        model_widgets = {}
        
        for scope, spec in model_specs.items():
            # Create widget
            widget = create_widget_from_spec(spec, self.on_change_callback)
            
            # Track it
            self.registry[scope] = widget
            model_widgets[scope] = widget
            
            # Add to form
            if hasattr(widget, 'get_widget'):
                # Add heading if needed
                heading_str = spec.get('group') or spec.get('heading')
                if heading_str and heading_str not in headings_added:
                    headings_added.add(heading_str)
                    hl = QLabel(f"{heading_str}")
                    hl.setStyleSheet("font-weight: bold; margin-top: 12px;")
                    param_form.addRow(hl)
                
                param_form.addRow(HoverLabel(spec['name'], spec.get('tooltip')), widget.get_widget())

        self.active_models_widgets[model_name] = model_widgets
        
        # Create Collapsible
        container_widget = QWidget()
        container_widget.setLayout(param_form)
        
        collapsible = CollapsibleWidget(
            model_name,
            container_widget,
            delete_callback=self.delete_model,
            deletable=True,
            is_visible=True
        )
        
        # Insert before the Adder
        layout = self.collapsible_container.scroll_content.layout()
        idx = layout.count() - 1 # Insert before last (Adder)
        layout.insertWidget(idx, collapsible)


    def add_model(self, model_class, model_name):
        """Callback for Add Button"""
        current_names = self.get_current_model_names()
        if model_name in current_names:
            warning_dialog(self, 'Invalid Name', f'Name `{model_name}` already exists and must be unique!')
            return
            
        # Generate default specs
        init_params = {model_name: {'name': model_name, self.plugin_class_key: model_class}} 
        model_config_list = self.plugin_factory.build_spec_from_user_config(init_params, update_default_values=True)
        print('\n\model_config_list specs')
        for k,v in model_config_list.items():
            for kk, vv in v.items():
                for kkk, vvv in vv.items():
                    print(kkk, vvv['current_value'])
            

        # Wrap in scope structure
        # Structure: Heading -> Keys -> default_value -> specs
        nested_structure = {
             self.plugin_heading: {
                 self.plugin_param_key: {
                     'default_value': None,
                     'current_value': model_config_list
                 }
             }
        }
        
        # Use interpreter to flatten to UI specs
        mInterp = SchemaInterpreter.from_specs(nested_structure, plugin_headings=[self.plugin_heading])
        new_specs = mInterp.get_ui_specs(unflatten=False)
        print('\n\nnew specs')
        for k,v in new_specs.items():
            print(k, v['current_value'])
        
        # Merge into our main specs
        self.plugin_specs.update(new_specs)
        print('\n\nplugin specs')
        for k,v in self.plugin_specs.items():
            print(k, v['current_value'])
        
        # Build UI
        self._build_model_ui(model_name, new_specs)
        
        # Signal change
        self.on_change_callback()


    def delete_model(self, model_name):
        """Callback for Delete Button on Collapsible"""

        print(f"Deleting model: {model_name}")
        from pprint import pprint
        print('active_models_widgets')
        pprint(self.active_models_widgets.keys())

        
        # Remove from UI handles
        if model_name in self.active_models_widgets:
            # Unregister specific widgets
            widgets_to_remove = self.active_models_widgets[model_name] # Model.MODEL_PARAMS.current_value.Stardist.root.default_reduce_fxn
            for scope in widgets_to_remove:
                if scope in self.registry:                             # Model.MODEL_PARAMS.current_value.Stardist.root.default_reduce_fxn
                    del self.registry[scope]
                else:
                    print(f"! scope not found in registry: {scope}")
                if scope in self.plugin_specs:                         # Model.MODEL_PARAMS.current_value.Stardist.root.default_reduce_fxn
                    del self.plugin_specs[scope]
                else:
                    print(f"! scope not found in plugin_specs: {scope}")
            
            del self.active_models_widgets[model_name]
        
        print('active_models_widgets')
        pprint(self.active_models_widgets.keys())
            
        # Trigger config changed
        self.on_change_callback()

    # --- Helpers ---

    def get_current_model_names(self):
        return list(self.active_models_widgets.keys())

    def _get_model_name_from_scope(self, scope):
        # e.g. 'Model.MODEL_PARAMS.current_value.stardist.root.param'
        # context: '{self.plugin_heading}.{self.plugin_param_key}.current_value.'
        prefix = f'{self.plugin_heading}.{self.plugin_param_key}.current_value.'
        if scope.startswith(prefix):
            remainder = scope[len(prefix):] # 'stardist.root.param'
            return remainder.split('.')[0]
        return None

    def _get_model_names_from_specs(self, specs):
        names = set()
        for scope in specs.keys():
            name = self._get_model_name_from_scope(scope)
            if name: names.add(name)
        return list(names)


# -----------------------------------------------------------------------------
# Main Widget
# -----------------------------------------------------------------------------

class ParamConfigWidget(QWidget):
    """
    Main Configuration Widget.
    Manages a tabbed interface of configuration parameters.
    Separates 'Static' configuration tabs from 'Plugin/Dynamic' configuration tabs.
    """
    built_config = pyqtSignal(name='builtConfig')
    config_changed = pyqtSignal(name='configChanged')

    def __init__(self, 
                 config_specs: dict, 
                 default_plugin_module_specs, 
                 plugin_param_map, 
                 plugin_class_key, 
                 plugin_factory, 
                 window_title='',
                 debug=False):
        
        super().__init__()
        self.setWindowTitle(window_title)
        self.debug = debug
        
        # Configuration
        self.initial_specs = config_specs
        self.default_plugin_module_specs = default_plugin_module_specs
        self.plugin_param_map = plugin_param_map # {'Model': 'MODEL_PARAMS'}
        self.plugin_class_key = plugin_class_key
        self.plugin_factory = plugin_factory
        
        # Identify Plugin Headings
        # We assume one plugin section for now based on original code usage, but design allows separation
        self.plugin_headings = list(plugin_param_map.keys())
        
        # State
        self.widget_registry = {} # Global map of {scope: widget_instance}
        self.config_data = {} # Last built config
        self.plugin_widgets_managers = {} # {Heading: PluginManagerWidget}
        
        # UI Setup
        self._init_layout()
        
        # Populate initial
        self.update_widget_layout(config_specs)


    def _init_layout(self):
        layout = QVBoxLayout()
        
        # Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("ConfigTabs")
        self.tab_widget.setStyleSheet("QTabBar::tab:selected {background-color: white; color: black;}")
        
        layout.addWidget(self.tab_widget)
        
        # Validation Button
        self.build_button = QPushButton("Validate Configuration")
        self.build_button.setStyleSheet("background-color: #D9D9D9; color: black;")
        self.build_button.clicked.connect(self.build_config)
        layout.addWidget(self.build_button)
        
        self.setLayout(layout)

    def _on_field_changed(self):
        """Slot for when any field changes."""
        self.build_button.setStyleSheet("background-color: #D9D9D9; color: black;")
        self.build_button.setText("Validate Configuration")
        self.config_changed.emit()

    def update_widget_layout(self, config_specs):
        """
        Populate the widget with specs.
        Or update existing widgets if they exist.
        """
        # 1. Separate Static vs Plugin specs
        static_groups = {} # {category: {scope: spec}}
        plugin_specs = {h: {} for h in self.plugin_headings} # {heading: {scope: spec}}
        
        for scope, spec in config_specs.items():
            # Get Category/Heading
            category_key = scope.split('.')[0]
            
            # Check if Plugin
            if category_key in self.plugin_headings:
                # The container param itself (e.g. Model.MODEL_PARAMS) is skipped 
                # because we reconstruct it from the children
                param_key = self.plugin_param_map[category_key]
                if scope == f"{category_key}.{param_key}":
                     continue
                
                plugin_specs[category_key][scope] = spec
                
            else:
                # Static Param
                if category_key not in static_groups:
                    static_groups[category_key] = {}
                static_groups[category_key][scope] = spec
        
        # 2. Handle Static Tabs
        # If tabs don't exist, create them. If they do, update values of widgets.
        for category, specs in static_groups.items():
            self._handle_static_category(category, specs)

        # 3. Handle Plugin Tabs
        for heading in self.plugin_headings:
            self._handle_plugin_category(heading, plugin_specs[heading])
            
            
    def _handle_static_category(self, category, specs):
        # Specific logic: if we are RE-updating, we assume structure is same and just set values.
        # But if widgets don't exist, we create tab.
        
        # Check if any widget in this group is missing
        missing = [s for s in specs if s not in self.widget_registry]
        
        if missing: 
            # If we have missing items, we likely are initializing or re-initializing the tab.
            # Simplified approach: Create/Recreate the tab.
            
            # Remove existing tab if exists (by name)
            self._remove_tab_by_name(category)
            
            # Create Group Widget
            group = StaticParamsGroup(category, specs, self.widget_registry, self._on_field_changed)
            
            # Wrap in Scroll Area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(group)
            
            self.tab_widget.addTab(scroll, category)
            
        else:
            # Update values
            for scope, spec in specs.items():
                val = spec.get('current_value')
                if val is not None:
                     self.widget_registry[scope].set_value(val)


    def _handle_plugin_category(self, heading, specs):
        # If manager exists, just update it. If not, create it.
        if heading not in self.plugin_widgets_managers:
            # Create Manager
            manager = PluginManagerWidget(
                plugin_heading=heading,
                plugin_param_key=self.plugin_param_map[heading],
                plugin_class_key=self.plugin_class_key,
                plugin_factory=self.plugin_factory,
                available_models=self.default_plugin_module_specs,
                registry=self.widget_registry,
                on_change_callback=self._on_field_changed,
                debug=self.debug
            )
            self.plugin_widgets_managers[heading] = manager
            
            # Add to Tab
            self.tab_widget.addTab(manager, heading)
            
            # Initially populate
            manager.update_from_specs(specs)
            
        else:
            # Just update specs (rebuilds internal list)
            self.plugin_widgets_managers[heading].update_from_specs(specs)


    def _remove_tab_by_name(self, name):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == name:
                self.tab_widget.removeTab(i)
                return

    # -------------------------------------------------------------------------
    # Public API (Used by MainApp)
    # -------------------------------------------------------------------------

    def set_widget_value(self, scope, value):
        if scope in self.widget_registry:
            self.widget_registry[scope].set_value(value)
        else:
            print(f"Warning: Attempted to set value for unknown scope: {scope}")

    def get_config(self):
        return self.config_data if self.config_data else None

    def get_current_model_names(self):
        # Aggregate from all plugin managers (though usually just one 'Model' manager)
        names = []
        for mgr in self.plugin_widgets_managers.values():
            names.extend(mgr.get_current_model_names())
        return names

    def build_config(self):
        """
        Compiles the current state of all widgets into a config dictionary.
            Iterate Registry - this covers 'Static' widgets directly AND 'Plugin' widgets.
            However, for Plugin widgets, need to reconstruct the nested dictionary structure that SchemaInterpreter expects.
        """
        
        print('\n\nbuilding config...')
        config = {}
        debug_str = ''
       
        # handle Plugin Managers (they hold the master specs for their items)
        for heading, mgr in self.plugin_widgets_managers.items():
            for scope, spec in mgr.plugin_specs.items():
                # Get current value from widget if possible
                if scope in self.widget_registry:
                    val = self.widget_registry[scope].get_value()
                    val = format_widget_value(val)
                    
                    # Update the spec object
                    spec['current_value'] = val
                    debug_str += f"{scope}: {val}\n"
                    
                    # Add to config as a SPEC
                    config[scope] = spec
                else:
                    print(f"Checking consistency: Scope {scope} in plugin specs but not in registry.")

        # handle Static Widgets - need to know which are static.
        plugin_scopes = set(config.keys()) # all just added are plugins
        
        for scope, widget in self.widget_registry.items():
            if scope in plugin_scopes:
                continue
            spec = self.initial_specs[scope]
            val = widget.get_value()
            val = format_widget_value(val)
            spec['current_value'] = val
            config[scope] = spec
            debug_str += f"{scope}: {val}\n"

        if self.debug:
            print(f"build_config:\n{debug_str}")
            
        self.config_data = config
        
        # Update UI state
        self.build_button.setStyleSheet("background-color: lightgreen; color: black;")
        self.build_button.setText("Configuration Built ✅")
        
        self.built_config.emit()
