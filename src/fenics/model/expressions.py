import asyncio
import uuid
from abc import abstractmethod

import XCore as xc
import XCoreHeadless
from s4l_core.simulator_plugins.base.model.controller_interface import TreeItem
from s4l_core.simulator_plugins.base.model.help import create_help_button, display_help
from XCorePython import PropertyExpression

EXPRESSION_INFO = """
- The spatial coordinate is defined as 'x' - to refer to the z position use a subscript: 'x[2]'.
- Most elementary nonlinear functions are available in the 'ufl' namespace (sin, exp, cos, bessel_J, ... ) e.g.: 'ufl.sinh(x[2])'
- The ufl namespace also contains a number of operators e.g dot, curl, grad, ...
See the ufl documentation for more details: https://fenics.readthedocs.io/projects/ufl/en/latest/
- The constant pi is defined e.g.: 'ufl.cos(2*pi*x[0])'
"""


class Expressions(TreeItem):
    def __init__(self, parent: TreeItem) -> None:
        super().__init__(parent, icon="icons/XCoreUI/Function.ico")

        self._properties = XCoreHeadless.DialogOptions()
        self._properties.Description = "Expressions"

        self._properties.Add("help_button", create_help_button())

        add_expression_prop = xc.PropertyPushButton("+")
        self._properties.Add("add", add_expression_prop)

        self._connect_signals()

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        asyncio.get_event_loop().call_soon(
            self._connect_signals
        )  # n.b. ensure de-pickling complete for all objects before trying to connect to other object's signals.

    def _connect_signals(self) -> None:
        help_button = self._properties.help_button
        assert isinstance(help_button, xc.PropertyPushButton)
        help_button.OnClicked.Connect(self._display_help)

        add_expression_prop = self._properties.add
        assert isinstance(add_expression_prop, xc.PropertyPushButton)

        add_expression_prop.OnClicked.Connect(self.add)

        # Connect up the delete buttons

        for child_group in self._properties.Children:
            if not isinstance(child_group, XCoreHeadless.DialogOptions):
                continue

            expr_delete = child_group.delete
            assert isinstance(expr_delete, xc.PropertyPushButton)

            def remove_term():
                child_group.Lose()

            expr_delete.OnClicked.Connect(remove_term)

    @abstractmethod
    def _display_help(self) -> None:
        """
        Calculate the help message and then display it.

        Should use display_help from the fenics_plugin.model.help module to display the help message
        """
        ...

    # n.b we do not allow the user to edit the description via the form
    @property
    def description(self) -> str:
        return self._properties.Description

    @description.setter
    def description(self, value: str) -> None:
        self._properties.Description = value

    @property
    def properties(self) -> xc.PropertyGroup:
        return self._properties

    def add(self, name: str = "my_expr", value: str = "42") -> None:
        expr_name = PropertyExpression(name)
        expr_value = PropertyExpression(value)
        expr_delete = xc.PropertyPushButton("Delete")
        expr_delete.Icon = "icons/XCoreUI/Delete.ico"

        expr_group = XCoreHeadless.DialogOptions()
        expr_group.Description = " "

        expr_group.Add("name", expr_name)
        expr_group.Add("value", expr_value)
        expr_group.Add("delete", expr_delete)

        self._properties.Add(str(uuid.uuid4()), expr_group)

        def remove_expr():
            expr_group.Lose()

        expr_delete.OnClicked.Connect(remove_expr)

    @property
    def expressions(self) -> dict[str, str]:
        exprs: dict[str, str] = {}

        for child_group in self._properties.Children:
            if not isinstance(child_group, XCoreHeadless.DialogOptions):
                continue

            name_prop = child_group.name
            assert isinstance(name_prop, PropertyExpression)

            value_prop = child_group.value
            assert isinstance(value_prop, PropertyExpression)

            exprs[name_prop.Value] = value_prop.Value

        return exprs


class GlobalExpressions(Expressions):
    def __init__(
        self,
        parent: TreeItem,
    ) -> None:
        super().__init__(parent)

        self._properties.Description = "Global Expressions"

    def _display_help(self) -> None:

        text = f"""
These expressions are available when defining coefficients in the subdomain or boundary settings.

The value of these expressions are evaluated in order (top - first) so expressions which have dependencies to other expressions should be defined last.

{EXPRESSION_INFO}
        """

        display_help("Global Expressions", text)


class PostProcessingExpressions(Expressions):
    def __init__(
        self,
        parent: TreeItem,
    ) -> None:
        super().__init__(parent)

        self._properties.Description = "Post-Processing Expressions"

    def _display_help(self) -> None:
        text = f"""
These expressions are evaluated after the dependent variables are calculated. They may result only in scalar or vector quantities. For example one may use them to evaluate the potential energy density following the evaluation of a potential field u as:

    energy_density: ufl.inner(ufl.grad(u), ufl.grad(u))

{EXPRESSION_INFO}
        """

        display_help("Post Processing Expressions", text)
