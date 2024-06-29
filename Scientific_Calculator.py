import cmath
import sys
import math
from math import log, exp, sin, cos, tan, radians, degrees, log10, isnan
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.integrate import odeint
import qdarkstyle
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def solve_cubic(a, b, c, d):
    if a == 0:
        raise ValueError("The coefficient 'a' cannot be zero for a cubic equation.")

    # Calculate discriminant
    delta0 = b ** 2 - 3 * a * c
    delta1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d
    discriminant = delta1 ** 2 - 4 * delta0 ** 3

    if discriminant > 0:
        # One real root, two complex roots
        C = ((delta1 + cmath.sqrt(discriminant)) / 2) ** (1 / 3)
        D = ((delta1 - cmath.sqrt(discriminant)) / 2) ** (1 / 3)
        real_root = -b / (3 * a) - (C + D)
        return [real_root]

    elif discriminant == 0:
        # All roots are real and at least two are equal
        if delta0 == 0:
            root = -b / (3 * a)
            return [root, root, root]
        else:
            K = delta1 / delta0
            root1 = -b / (3 * a) + K
            root2 = -K / 2
            return [root1, root2]

    else:
        # All roots are real and distinct
        C = ((delta1 + cmath.sqrt(discriminant)) / 2) ** (1 / 3)
        D = ((delta1 - cmath.sqrt(discriminant)) / 2) ** (1 / 3)
        root1 = -b / (3 * a) + (C + D)
        real_part = -(C + D) / 2
        imaginary_part = (C - D) * cmath.sqrt(3) / 2
        root2 = -(b + real_part * a) / (3 * a) + imaginary_part
        root3 = -(b + real_part * a) / (3 * a) - imaginary_part
        return [root1, root2, root3]

class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Cubic Equation Solver')
        self.setGeometry(500, 500, 500, 500)

        layout = QVBoxLayout()

        self.coefficient_labels = ['a(Coeff. of x^3)', 'b(Coeff. of x^2)', 'c(Coeff. of x)', 'd(Constant term)']
        self.coefficient_lineedits = []

        for label_text in self.coefficient_labels:
            label = QLabel(label_text)
            layout.addWidget(label)
            lineedit = QLineEdit()
            layout.addWidget(lineedit)
            self.coefficient_lineedits.append(lineedit)

        solve_button = QPushButton('Solve')
        solve_button.clicked.connect(self.solveClicked)
        layout.addWidget(solve_button)

        clear_button = QPushButton('Clear')
        clear_button.clicked.connect(self.clearClicked)
        layout.addWidget(clear_button)

        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def solveClicked(self):
        coefficients = []
        for lineedit in self.coefficient_lineedits:
            try:
                coefficient = float(lineedit.text())
                coefficients.append(coefficient)
            except ValueError:
                QMessageBox.warning(self, 'Warning', 'Please enter valid coefficients.')
                return

        if len(coefficients) != 4:
            QMessageBox.warning(self, 'Warning', 'Please enter all coefficients.')
            return

        try:
            roots = solve_cubic(*coefficients)
            roots_str = ', '.join([f'{root:.2f}' for root in roots])
            self.result_label.setText(f'Roots of the cubic equation are: {roots_str}')
        except ValueError as e:
            QMessageBox.warning(self, 'Error', str(e))

    def clearClicked(self):
        for lineedit in self.coefficient_lineedits:
            lineedit.clear()
        self.result_label.clear()

class NewtonInterpolationCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Newton Interpolation Calculator")
        self.setGeometry(250, 350, 500, 300)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.data_points_layout = QVBoxLayout()
        self.add_data_point_button = QPushButton("Add Data Point")
        self.add_data_point_button.clicked.connect(self.add_data_point)
        layout.addLayout(self.data_points_layout)
        layout.addWidget(self.add_data_point_button)

        self.x_interp_edit = QLineEdit()
        self.x_interp_edit.setPlaceholderText("Enter x for interpolation")
        self.interpolated_value_label = QLabel("Interpolated Value:")
        layout.addWidget(self.x_interp_edit)
        layout.addWidget(self.interpolated_value_label)

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_interpolation)
        layout.addWidget(self.calculate_button)

        self.setLayout(layout)

        self.data_points = []

    def add_data_point(self):
        data_point_layout = QHBoxLayout()
        x_edit = QLineEdit()
        y_edit = QLineEdit()
        data_point_layout.addWidget(QLabel("x:"))
        data_point_layout.addWidget(x_edit)
        data_point_layout.addWidget(QLabel("y:"))
        data_point_layout.addWidget(y_edit)
        self.data_points_layout.addLayout(data_point_layout)
        self.data_points.append((x_edit, y_edit))

    def calculate_interpolation(self):
        x_values = []
        y_values = []
        for x_edit, y_edit in self.data_points:
            x_value = float(x_edit.text())
            y_value = float(y_edit.text())
            x_values.append(x_value)
            y_values.append(y_value)

        x_interp = float(self.x_interp_edit.text())

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        interpolated_value = self.newton_interpolation(x_values, y_values, x_interp)
        self.interpolated_value_label.setText(f"Interpolated Value: {interpolated_value}")

    def newton_interpolation(self, x, y, x_interp):
        coef = self.divided_difference(x, y)
        n = len(x)
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = result * (x_interp - x[i]) + coef[i]
        return result

    def divided_difference(self, x, y):
        n = len(y)
        coef = np.copy(y)
        for j in range(1, n):
            coef[j:] = (coef[j:] - coef[j - 1:-1]) / (x[j:] - x[:-j])
        return coef

class QuadraticEquationSolver(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Quadratic Equation Solver')
        self.setWindowIcon(QIcon(r"C:\Users\HP\Desktop\scientific-calculator.png"))
        self.setGeometry(100, 100, 400, 250)
        self.setStyleSheet("background-color: #232323; color: white;")

        # Header label with FontAwesome icon
        header_label = QLabel(self)
        header_label.setText("<font size='5' color='#4CAF50'><i class='fab fa-fort-awesome'></i></font> <font size='5'>Quadratic Equation Solver</font>")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Labels for coefficients
        label_a = QLabel('Coefficient a(x^2):', self)
        label_b = QLabel('Coefficient b(x):', self)
        label_c = QLabel('Coefficient c:', self)

        # Input fields for coefficients
        self.a_input = QLineEdit(self)
        self.b_input = QLineEdit(self)
        self.c_input = QLineEdit(self)

        # Label to display the result
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Solve button with hover effect
        solve_button = QPushButton('Solve', self)
        solve_button.clicked.connect(self.solve_quadratic)
        solve_button.setStyleSheet('QPushButton {background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px;}'
                                    'QPushButton:hover {background-color: #45a049;}')

        # Set size policy for each widget to expand to fill available space
        header_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        label_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.a_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        label_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.b_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        label_c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.c_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        solve_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Create layouts
        input_layout = QVBoxLayout()
        input_layout.addWidget(label_a)
        input_layout.addWidget(self.a_input)
        input_layout.addWidget(label_b)
        input_layout.addWidget(self.b_input)
        input_layout.addWidget(label_c)
        input_layout.addWidget(self.c_input)

        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(solve_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(header_label)
        main_layout.addLayout(input_layout)
        main_layout.addLayout(result_layout)
        main_layout.addLayout(button_layout)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def solve_quadratic(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            c = float(self.c_input.text())

            # Calculate the discriminant
            discriminant = b**2 - 4*a*c

            if discriminant > 0:
                root1 = (-b + discriminant**0.5) / (2*a)
                root2 = (-b - discriminant**0.5) / (2*a)
                result = f"Roots: {root1:.2f}, {root2:.2f}"
            elif discriminant == 0:
                root1 = root2 = -b / (2*a)
                result = f"Roots: {root1:.2f}"
            else:
                real_part = -b / (2*a)
                imaginary_part = (-discriminant)**0.5 / (2*a)
                result = f"Roots: {real_part:.2f} + {imaginary_part:.2f}i, {real_part:.2f} - {imaginary_part:.2f}i"

            self.animate_result(result)

        except ValueError:
            self.show_error("Invalid Input", "Please enter valid numerical coefficients.")

    def show_error(self, title, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.exec_()

    @pyqtSlot(str)
    def animate_result(self, result):
        self.result_label.setText(result)
        animation = QPropertyAnimation(self.result_label, b'opacity')
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setDuration(1000)

class BasicCalculator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Basic Calculator")
        self.setWindowIcon(QIcon(r"C:\Users\HP\Desktop\scientific-calculator.png"))
        self.setGeometry(100, 100, 300, 400)

        # Set dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #282828;
                color: #f8f8f2;
            }
            QPushButton {
                background-color: #3c3836;
                color: #f8f8f2;
                border: none;
            }
            QPushButton:hover {
                background-color: #45413b;
            }
            QLineEdit {
                background-color: #3c3836;
                color: #f8f8f2;
                border: none;
            }
        """)

        # Create the main layout
        main_layout = QVBoxLayout()

        # Create the display widget
        self.result_display = QLineEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_layout.addWidget(self.result_display)

        # Create the button grid layout
        button_grid_layout = QVBoxLayout()

        #Row 0
        row0 = self.create_button_row(['', '', ' ', 'C'])
        button_grid_layout.addLayout(row0)

        # Row 1
        row1 = self.create_button_row(['7', '8', '9', '/'])
        button_grid_layout.addLayout(row1)

        # Row 2
        row2 = self.create_button_row(['4', '5', '6', '*'])
        button_grid_layout.addLayout(row2)

        # Row 3
        row3 = self.create_button_row(['1', '2', '3', '-'])
        button_grid_layout.addLayout(row3)

        # Row 4
        row4 = self.create_button_row(['0', '.', '=', '+'])
        button_grid_layout.addLayout(row4)

        # Add the button grid layout to the main layout
        main_layout.addLayout(button_grid_layout)

        # Set the main layout for the widget
        self.setLayout(main_layout)

    def create_button_row(self, button_texts):
        row_layout = QHBoxLayout()

        for button_text in button_texts:
            button = QPushButton(button_text)
            button.clicked.connect(self.on_button_click)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row_layout.addWidget(button)

        return row_layout

    def on_button_click(self):
        button = self.sender()
        current_text = self.result_display.text()

        if button.text() == "=":
            try:
                result = str(eval(current_text))
                self.result_display.setText(result)
            except Exception as e:
                self.result_display.setText("Error")
        elif button.text() == 'C':
            self.result_display.clear()
        else:
            self.result_display.setText(current_text + button.text())


    def on_clear_click(self):
        self.result_display.clear()

def euler_method(f, x_lower, x_upper, y_initial, step_size):
    x_values = [x_lower]
    y_values = [y_initial]
    
    x_current = x_lower
    y_current = y_initial
    
    while x_current + step_size <= x_upper:
        slope = f(x_current, y_current)
        x_current += step_size
        y_current += step_size * slope
        x_values.append(x_current)
        y_values.append(y_current)
    
    if x_current < x_upper:
        step_size = x_upper - x_current
        slope = f(x_current, y_current)
        x_current += step_size
        y_current += step_size * slope
        x_values.append(x_current)
        y_values.append(y_current)

            
    return x_values, y_values

class ODESolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ODE Solver with Euler Method')
        self.setGeometry(100, 100, 600, 500)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        
        layout = QVBoxLayout()
        
        form_layout = QVBoxLayout()
        self.x_lower_input = self.create_input_field("Lower value of x:", form_layout)
        self.x_upper_input = self.create_input_field("Upper value of x:", form_layout)
        self.y_initial_input = self.create_input_field("Initial value of y:", form_layout)
        self.step_size_input = self.create_input_field("Step size:", form_layout)
        self.function_input = self.create_input_field("Function dy/dx (e.g., 'x + y'):", form_layout)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        calc_button = QPushButton('Calculate')
        calc_button.clicked.connect(self.on_calculate)

        layout.addLayout(form_layout)
        layout.addWidget(calc_button)
        layout.addWidget(self.result_text)

        centralWidget.setLayout(layout)

    def create_input_field(self, label, layout):
        row = QHBoxLayout()
        label = QLabel(label)
        input_field = QLineEdit()
        row.addWidget(label)
        row.addWidget(input_field)
        layout.addLayout(row)
        return input_field

    @pyqtSlot()
    def on_calculate(self):
        try:
            # Get input values
            x_lower = float(self.x_lower_input.text())
            x_upper = float(self.x_upper_input.text())
            y_initial = float(self.y_initial_input.text())
            step_size = float(self.step_size_input.text())
            function_text = self.function_input.text()

            # Validate input values
            if not all([x_lower, x_upper, y_initial, step_size, function_text]):
                raise ValueError("All input fields must be filled.")

            # Evaluate the function
            eval_env = {"sin": math.sin, "cos": math.cos, "exp": math.exp, "log": math.log, "y": y_initial}
            f = eval("lambda x, y: " + function_text, {"__builtins__": None}, eval_env)

            # Perform Euler method
            x_values, y_values = euler_method(f, x_lower, x_upper, y_initial, step_size)

            # Display results
            result_str = '\n'.join(f"x = {x:.4f}, y = {y:.4f}" for x, y in zip(x_values, y_values))
            self.result_text.setText(result_str)

        except Exception as e:
            self.result_text.setText(f"Error: {e}")

class TranscendentalEquationSolver(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Widgets
        self.equation_label = QLabel("Enter the transcendental equation (use 'x' as the variable):")
        self.equation_input = QLineEdit(self)

        self.guess_label = QLabel("Enter an initial guess for the root:")
        self.guess_input = QDoubleSpinBox(self)
        self.guess_input.setRange(-1000, 1000)
        self.guess_input.setSingleStep(0.1)

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)

        self.solve_button = QPushButton("Solve", self)
        self.solve_button.clicked.connect(self.solve)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.equation_label)
        layout.addWidget(self.equation_input)
        layout.addWidget(self.guess_label)
        layout.addWidget(self.guess_input)
        layout.addWidget(self.solve_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Transcendental Equation Solver')

        # Apply qdarkstyle globally
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    def solve(self):
        equation_str = self.equation_input.text()
        x = sp.symbols('x')
        equation = sp.sympify(equation_str)
        derivative = sp.diff(equation, x)
        x0 = self.guess_input.value()

        max_iterations = 20
        tolerance = 1e-6

        for i in range(max_iterations):
            f_x = equation.subs(x, x0)
            f_prime_x = derivative.subs(x, x0)
            x0 = x0 - f_x.evalf() / f_prime_x.evalf()

            self.result_text.append(f"Iteration {i + 1}: x = {x0}")

            if sp.N(f_x.evalf()).is_zero:
                self.result_text.append(f"Converged to root: {x0}")
                break
        else:
            self.result_text.append("Maximum number of iterations reached. Solution may not have converged.")

    def clear(self):
        # Clear the input fields and result text
        self.guess_input.clear()
        self.result_text.clear()

class NumericalIntegration(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numerical Integration")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.func_label = QLabel("Enter the function to integrate (e.g., 'np.sin(x)', 'x**2', etc.):")
        self.func_input = QLineEdit()
        layout.addWidget(self.func_label)
        layout.addWidget(self.func_input)

        self.lower_label = QLabel("Enter the lower limit of integration:")
        self.lower_input = QLineEdit()
        layout.addWidget(self.lower_label)
        layout.addWidget(self.lower_input)

        self.upper_label = QLabel("Enter the upper limit of integration:")
        self.upper_input = QLineEdit()
        layout.addWidget(self.upper_label)
        layout.addWidget(self.upper_input)

        self.steps_label = QLabel("Enter the number of subintervals (step size):")
        self.steps_input = QLineEdit()
        layout.addWidget(self.steps_label)
        layout.addWidget(self.steps_input)

        self.calculate_button = QPushButton("Calculate Integral")
        self.calculate_button.clicked.connect(self.calculate_integral)
        layout.addWidget(self.calculate_button)

        self.clear_button = QPushButton("Clear Input")
        self.clear_button.clicked.connect(self.clear_input)
        layout.addWidget(self.clear_button)

        self.result_label = QLabel("Result will be shown here")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def parse_function(self, func_str):
        def func(x):
            return eval(func_str)
        return func

    def trapezoidal_rule(self, f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    def simpsons_1_3_rule(self, f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        return h/3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

    def simpsons_3_8_rule(self, f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        result = y[0] + y[-1]
        for i in range(1, n):
            weight = 3 if i % 3 != 0 else 2
            result += weight * y[i]
        return 3 * h / 8 * result

    def calculate_integral(self):
        func_str = self.func_input.text()
        a = float(self.lower_input.text())
        b = float(self.upper_input.text())
        n = int(self.steps_input.text())

        f = self.parse_function(func_str)

        if n % 3 == 0:
            result = self.simpsons_3_8_rule(f, a, b, n)
            method = "Simpson's 3/8 rule"
        elif n % 2 == 0:
            result = self.simpsons_1_3_rule(f, a, b, n)
            method = "Simpson's 1/3 rule"
        else:
            result = self.trapezoidal_rule(f, a, b, n)
            method = "Trapezoidal rule"

        self.result_label.setText(f"The integral of the function from {a} to {b} using {method} is approximately {result}")

    def clear_input(self):
        self.func_input.clear()
        self.lower_input.clear()
        self.upper_input.clear()
        self.steps_input.clear()
        self.result_label.setText("Result will be shown here")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowIcon(QIcon(r"C:\Users\HP\Desktop\pen.png"))
        MainWindow.resize(800, 700)
        MainWindow.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 731, 101))
        font = QtGui.QFont()
        font.setFamily("Perpetua Titling MT")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.qes = QtWidgets.QPushButton(self.centralwidget)
        self.qes.setGeometry(QtCore.QRect(90, 190, 231, 51))
        self.qes.setObjectName("qes")
        self.qes.clicked.connect(self.show_quadratic_solver)
        self.sc = QtWidgets.QPushButton(self.centralwidget)
        self.sc.setGeometry(QtCore.QRect(490, 190, 231, 51))
        self.sc.setObjectName("sc")
        self.sc.clicked.connect(self.show_basic_calculator)
        self.let = QtWidgets.QPushButton(self.centralwidget)
        self.let.setGeometry(QtCore.QRect(90, 320, 231, 51))
        self.let.setObjectName("let")
        self.let.clicked.connect(self.show_Calculator)
        self.ode = QtWidgets.QPushButton(self.centralwidget)
        self.ode.setGeometry(QtCore.QRect(490, 320, 231, 51))
        self.ode.setObjectName("ode")
        self.ode.clicked.connect(self.show_ODESolverApp)
        self.tes = QtWidgets.QPushButton(self.centralwidget)
        self.tes.setGeometry(QtCore.QRect(90, 450, 231, 51))
        self.tes.setObjectName("tes")
        self.tes.clicked.connect(self.show_TranscendentalEquationSolver)
        self.nic = QtWidgets.QPushButton(self.centralwidget)
        self.nic.setGeometry(QtCore.QRect(490, 450, 231, 51))
        self.nic.setObjectName("nic")
        self.nic.clicked.connect(self.show_NewtonInterpolationCalculator)
        self.ni = QtWidgets.QPushButton(self.centralwidget)
        self.ni.setGeometry(QtCore.QRect(90, 570, 231, 51))
        self.ni.setObjectName("ni")
        self.ni.clicked.connect(self.show_NumericalIntegration)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Scientific Computing Calculator"))
        self.label.setText(_translate("MainWindow", "SCIENTIFIC COMPUTING"))
        self.qes.setText(_translate("MainWindow", "Quadratic Equation Solver"))
        self.let.setText(_translate("MainWindow", "Cubic Equation Solver"))
        self.sc.setText(_translate("MainWindow", "Standard Calculator"))
        self.ode.setText(_translate("MainWindow", "ODE Solver"))
        self.tes.setText(_translate("MainWindow", "Transcendental Equation Solver"))
        self.nic.setText(_translate("MainWindow", "Newton Interpolation "))
        self.ni.setText(_translate("MainWindow", "Numerical Integration"))

    def show_quadratic_solver(self):
        self.quadratic_solver = QuadraticEquationSolver()
        self.quadratic_solver.show()

    def show_basic_calculator(self):
        self.basic_calculator = BasicCalculator()
        self.basic_calculator.show()

    def show_Calculator(self):
        self.Calculator = Calculator()
        self.Calculator.show()

    def show_ODESolverApp(self):
        self.ODESolverApp = ODESolverApp()
        self.ODESolverApp.show()

    def show_TranscendentalEquationSolver(self):
        self.TranscendentalEquationSolver = TranscendentalEquationSolver()
        self.TranscendentalEquationSolver.show()

    def show_NewtonInterpolationCalculator(self):
        self.NewtonInterpolationCalculator = NewtonInterpolationCalculator()
        self.NewtonInterpolationCalculator.show()

    def show_NumericalIntegration(self):
        self.NumericalIntegration = NumericalIntegration()
        self.NumericalIntegration.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())