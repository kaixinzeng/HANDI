import numpy as np
import pandas as pd
import pysindy as ps
from tqdm import tqdm
import os
import re
from sympy import sympify, expand, Symbol
from typing import List, Dict, Tuple, Optional

def linear_system(x,t):
    x1, x2 = x[0], x[1]
    return np.stack([0.1 * x1 + 3.0 * x2, -3.0 * x1 + 0.1 * x2])

def duffing_system(x,t):
    x1, x2 = x[0], x[1]
    return np.stack([1.0 * x2, 5.0 *x1 - 0.5 * x2 - 1.0 * x1 * x1 * x1])

def fixed_point_system(x,t):
    x1, x2 = x[0], x[1]
    r2 = x1**2 + x2**2
    return np.stack([-3.0 * x2 - x1 * r2, 3.0 * x1 - x2 * r2])

def limit_cycle_system(x,t):
    x1, x2 = x[0], x[1]
    r2 = x1**2 + x2**2
    return np.stack([3.0 * x2 - x1 * (r2 - 1), -3.0 * x1 - x2 * (r2 - 1)])

def vdp(x,t, mu=1.0):
    x1, x2 = x[0], x[1]
    dxdt = x2
    d2xdt2 = mu * (1 - x1**2) * x2 - x1
    return  np.stack([dxdt, d2xdt2])

def m1k16c8_system(x,t):
    x1, x2 = x[0], x[1]
    return  np.stack([x2, -16.0 * x1 - 8.0 * x2])

def m1k17c2_system(x,t):
    x1, x2 = x[0], x[1]
    return  np.stack([x2, -17.0 * x1 - 2.0 * x2])

def m1k25c20_system(x,t):
    x1, x2 = x[0], x[1]
    return  np.stack([x2, -25.0 * x1 - 20.0 * x2])

def downsample(dt,data):
    data_length=len(data)
    step_ratio = dt / 0.01
    step = int(round(step_ratio))
    indices = np.arange(0, data_length, step)
    t_target = indices * 0.01
    data_downsampled = data[indices]
    time_error = np.max(np.abs(t_target - indices * 0.01))
    if time_error > 1e-12:
        print(f"Warning: large time error for dt={dt}: {time_error:.2e}")
    return data_downsampled

def replace_dot_to_mul(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("·", "*")
    out_file = filepath.replace(".txt", "_mul.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    return out_file


def create_learned_model(equations):
    def model(t, x):
        dxdt = np.zeros(len(equations), dtype=float)
        for i, eq in enumerate(equations):
            try:
                val = eval(eq, {"np": np, "x": x}) 
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                dxdt[i] = float(val)
            except Exception:
                dxdt[i] = 0.0
        return dxdt
    return model

def parse_and_prepare_equations(file, dim):
    try:
        with open(file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None

    pattern = r"dx\((\d+)\)/dt = (.*?)(?=\ndx\(\d+\)|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    equations = {}
    for var_idx_str, eq in matches:
        eq_parsed = re.sub(r'x\((\d+)\)', lambda m: f'x[{int(m.group(1)) - 1}]', eq)
        equations[int(var_idx_str) - 1] = eq_parsed.strip()

    equations_list = [equations.get(i, "0.0") for i in range(dim)]
    
    if len(equations_list) > dim:
        equations_list = equations_list[:dim]
        
    return equations_list

def compute_r2(y_true,y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2

def change_equations(file,output_file):
    with open(file, 'r') as f:
        content = f.read()

        pattern = r'dx0/dt\s*=\s*(.+?)\s*\\?\n?\s*dx1/dt\s*=\s*(.+?)$'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)

        if match:
            eq1_str = match.group(1).strip()
            eq2_str = match.group(2).strip()

            x0, x1 = Symbol('x0'), Symbol('x1')

            expr1 = expand(sympify(eq1_str))
            expr2 = expand(sympify(eq2_str))
            eq1_expanded = str(expr1)
            eq2_expanded = str(expr2)

            eq1_final = eq1_expanded
            eq2_final = eq2_expanded
            def replace_power(match):
                var = match.group(1)
                power = int(match.group(2))
                if power == 0:
                    return "1"
                elif power == 1:
                    return var
                else:
                    return "*".join([var] * power)
            pattern = r'([a-zA-Z_]\w*)\*\*(\d+)'
            eq1_final = re.sub(pattern, replace_power, eq1_final)
            eq2_final = re.sub(pattern, replace_power, eq2_final)
            eq1_final = eq1_final.replace('x0', 'x(1)').replace('x1', 'x(2)')
            eq2_final = eq2_final.replace('x0', 'x(1)').replace('x1', 'x(2)')

    with open(output_file, 'w') as f:
        f.write(f"dx(1)/dt = {eq1_final}\n")
        f.write(f"dx(2)/dt = {eq2_final}\n")

def compute_nrmse(w_true, w_pred):
    """Computes the Normalized Root Mean Squared Error for the coefficients."""
    #w_true_flat = w_true.flatten()
    #w_pred_flat = w_pred.flatten()
    w_pred = w_pred
    rmse = np.sqrt(np.mean((w_true - w_pred) ** 2))
    nonzero_mask = w_true != 0
    if not np.any(nonzero_mask):
        return rmse
    avg_abs_w = np.mean(np.abs(w_true[nonzero_mask]))
    return rmse / avg_abs_w

def read_equations_from_file(filepath: str) -> str:
    """Read all lines from a .txt file and return as a single string."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    
def parse_equations_to_coeff_dict(equation_text: str, dim: int) -> Dict[Tuple[int, ...], List[float]]:
    coeff_dict = {}
    lines = [line.strip() for line in equation_text.strip().split('\n') if line.strip()]
    
    for line in lines:
        match = re.match(r'dx\((\d+)\)/dt\s*=\s*(.+)', line)
        if not match:
            continue
        var_idx = int(match.group(1)) - 1
        if var_idx >= dim:
            continue
        expr = match.group(2)

        # Normalize
        expr = expr.replace('−', '-').replace(' ', '')
        if expr.startswith('-'):
            expr = '0+' + expr
        expr = expr.replace('-', '+-')
        terms = [t for t in expr.split('+') if t]

        for term in terms:
            if not term:
                continue

            # Try to extract coefficient
            coef_match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\*(.+)$', term)
            if coef_match:
                try:
                    coef = float(coef_match.group(1))
                    var_part = coef_match.group(2)
                except ValueError:
                    continue
            else:
                if term.startswith(('+', '-')):
                    sign = -1.0 if term.startswith('-') else 1.0
                    var_part = term[1:]
                    coef = sign
                else:
                    var_part = term
                    coef = 1.0

            exp_tuple = _parse_var_part_to_exponents(var_part, dim)
            if exp_tuple is None:
                continue

            if exp_tuple not in coeff_dict:
                coeff_dict[exp_tuple] = [0.0] * dim
            coeff_dict[exp_tuple][var_idx] += coef

    return coeff_dict

def _parse_var_part_to_exponents(var_part: str, dim: int) -> Optional[Tuple[int, ...]]:
    matches = re.findall(r'x\((\d+)\)', var_part)
    if not matches:
        return tuple([0] * dim)
    exponents = [0] * dim
    for m in matches:
        idx = int(m) - 1
        if idx < dim:
            exponents[idx] += 1
        else:
            return None
    return tuple(exponents)

def exponent_tuple_to_feature_name(exp_tuple: Tuple[int, ...]) -> str:
    parts = []
    for i, exp in enumerate(exp_tuple):
        if exp == 0:
            continue
        var = f"x{i+1}"
        if exp == 1:
            parts.append(var)
        else:
            parts.append(f"{var}^{exp}")
    return "1" if not parts else " ".join(parts)

def build_true_coeff_from_file(
    filepath: str,
    names: List[str],
    dim: int
) -> np.ndarray:
    """
    Read equations from .txt file and return true_coeff of shape (len(names), dim)
    """
    equation_text = read_equations_from_file(filepath)
    coeff_dict = parse_equations_to_coeff_dict(equation_text, dim)

    name_to_coeffs = {}
    for exp_tuple, coeffs in coeff_dict.items():
        feat_name = exponent_tuple_to_feature_name(exp_tuple)
        name_to_coeffs[feat_name] = coeffs

    num_features = len(names)
    true_coeff = np.zeros((num_features, dim))
    for i, name in enumerate(names):
        if name in name_to_coeffs:
            true_coeff[i, :] = name_to_coeffs[name]
    return true_coeff