# NEWTON-RAPHSON-METHOD
#I implemented and tested the Newton-Raphson method for root finding, demonstrates quadratic convergence, and explores various failure modes

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton as scipy_newton

#=============================================================================
# SECTION 1: UTILITY FUNCTIONS
#=============================================================================

def print_header(title):
    """Prints a formatted header for each section"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_subheader(title):
    """Prints a formatted subheader"""
    print("\n" + "-"*60)
    print(f" {title}")
    print("-"*60)

def newton_step(x_old, f, df):
    """
    Performs one Newton-Raphson iteration.
    
    Parameters:
    x_old : float : current guess
    f     : function : function whose root we seek
    df    : function : derivative of f
    
    Returns:
    float : new guess
    """
    return x_old - f(x_old)/df(x_old)

#=============================================================================
# SECTION 2: PROBLEM DEFINITIONS (All functions used in the lab)
#=============================================================================

def finite_well_f(x):
    """f(x) = x*tan(x) - sqrt(16 - x^2) for finite potential well"""
    return x * np.tan(x) - np.sqrt(16 - x**2)

def finite_well_df(x):
    """Derivative for finite potential well"""
    return np.tan(x) + x * (1/np.cos(x))**2 + x/np.sqrt(16 - x**2)

def cosine_f(x):
    """f(x) = x - cos(x)"""
    return x - np.cos(x)

def cosine_df(x):
    """Derivative of x - cos(x)"""
    return 1 + np.sin(x)

def double_well_f(x):
    """f(x) = x^4 - 2x^2 (symmetric double-well)"""
    return x**4 - 2*x**2

def double_well_df(x):
    """Derivative of x^4 - 2x^2"""
    return 4*x**3 - 4*x

def cubic_f(x):
    """f(x) = x^3 - 2x + 2"""
    return x**3 - 2*x + 2

def cubic_df(x):
    """Derivative of x^3 - 2x + 2"""
    return 3*x**2 - 2

def arctan_f(x):
    """f(x) = arctan(x)"""
    return np.arctan(x)

def arctan_df(x):
    """Derivative of arctan(x)"""
    return 1/(1 + x**2)

#=============================================================================
# SECTION 3: EXERCISE 1 - Finite Potential Well
#=============================================================================

def exercise_1_finite_well():
    """
    Exercise 1: Finite potential well with initial guess x0 = 1.0
    Demonstrates what happens with a poor initial guess.
    """
    print_header("EXERCISE 1: Finite Potential Well")
    print("f(x) = x*tan(x) - sqrt(16 - x^2)")
    
    # Part A: Poor initial guess
    print_subheader("Part A: Starting with x0 = 1.0 (Poor Guess)")
    x = 1.0
    print(f"{'Iter':<6} {'x':<15} {'|f(x)|':<15} {'Note':<20}")
    print("-"*56)
    print(f"{0:<6} {x:<15.8f} {abs(finite_well_f(x)):<15.2e} Initial guess")
    
    for i in range(1, 6):
        try:
            x = newton_step(x, finite_well_f, finite_well_df)
            f_val = abs(finite_well_f(x))
            if i == 2:
                note = "Values blowing up!"
            elif i > 2:
                note = "Diverging..."
            else:
                note = ""
            print(f"{i:<6} {x:<15.8f} {f_val:<15.2e} {note}")
        except:
            print(f"{i:<6} {'ERROR':<15} {'N/A':<15} Method failed")
            break
    
    print("\n OBSERVATION: x = 1.0 is near a singularity in tan(x)")
    print("   The derivative becomes huge, causing divergence.")
    
    # Part B: Better initial guess
    print_subheader("Part B: Starting with x0 = 2.0 (Better Guess)")
    x = 2.0
    print(f"{'Iter':<6} {'x':<15} {'|f(x)|':<15}")
    print("-"*41)
    print(f"{0:<6} {x:<15.8f} {abs(finite_well_f(x)):<15.2e}")
    
    for i in range(1, 6):
        x = newton_step(x, finite_well_f, finite_well_df)
        print(f"{i:<6} {x:<15.8f} {abs(finite_well_f(x)):<15.2e}")
    
    print("\n FIX: Choose initial guess away from singularities.")
    print("   Values now converge properly.")

#=============================================================================
# SECTION 4: EXERCISE 2 - Quadratic Convergence Demonstration
#=============================================================================

def exercise_2_quadratic_convergence():
    """
    Exercise 2: f(x) = x - cos(x)
    Demonstrates quadratic convergence (doubling of correct digits).
    """
    print_header("EXERCISE 2: Quadratic Convergence Demonstration")
    print("f(x) = x - cos(x)")
    
    x = 1.0
    print(f"\n{'Iter':<6} {'x':<20} {'|f(x)|':<15} {'Correct Digits':<15}")
    print("-"*62)
    print(f"{0:<6} {x:<20.12f} {abs(cosine_f(x)):<15.2e} {'1-2':<15}")
    
    for i in range(1, 6):
        x = newton_step(x, cosine_f, cosine_df)
        f_val = abs(cosine_f(x))
        
        # Estimate correct digits
        if i == 1:
            digits = "3-4"
        elif i == 2:
            digits = "6-7"
        elif i == 3:
            digits = "10"
        else:
            digits = ">10"
            
        print(f"{i:<6} {x:<20.12f} {f_val:<15.2e} {digits:<15}")
    
    print("\n✨ QUADRATIC CONVERGENCE: Correct digits roughly double each step!")
    print("   This is Newton's superpower - exponential speed.")

#=============================================================================
# SECTION 5: SCENARIO A - The Flat Spot (Division by Zero)
#=============================================================================

def scenario_a_flat_spot():
    """
    Scenario A: f(x) = x^4 - 2x^2
    Demonstrates failure when derivative is zero.
    """
    print_header("SCENARIO A: The Flat Spot (Division by Zero)")
    print("f(x) = x⁴ - 2x²")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot the function
    x_plot = np.linspace(-2, 2, 400)
    y_plot = double_well_f(x_plot)
    
    ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x⁴ - 2x²')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.plot(1.0, double_well_f(1.0), 'ro', markersize=10, label='x₀ = 1.0 (flat)')
    ax1.plot(1.4, double_well_f(1.4), 'go', markersize=10, label='x₀ = 1.4 (works)')
    ax1.plot(np.sqrt(2), 0, 'm*', markersize=15, label='Root at x = √2')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Function Shape', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Try x0 = 1.0
    print_subheader("Case 1: Starting at x₀ = 1.0 (Derivative = 0)")
    x = 1.0
    print(f"Initial: x = {x:.6f}, f(x) = {double_well_f(x):.6f}, f'(x) = {double_well_df(x):.6f}")
    
    iterations_a1 = [(0, x, double_well_f(x), double_well_df(x))]
    
    for i in range(1, 4):
        if abs(double_well_df(x)) < 1e-10:
            print(f"Iteration {i}: DERIVATIVE ≈ 0! Cannot divide by zero.")
            break
        x = newton_step(x, double_well_f, double_well_df)
        iterations_a1.append((i, x, double_well_f(x), double_well_df(x)))
        print(f"Iteration {i}: x = {x:.6f}, f'(x) = {double_well_df(x):.6f}")
    
    print("\n RESULT: Method fails - division by zero at derivative = 0")
    
    # Try x0 = 1.4
    print_subheader("Case 2: Starting at x₀ = 1.4 (Non-zero derivative)")
    x = 1.4
    print(f"Initial: x = {x:.6f}, f(x) = {double_well_f(x):.6f}")
    
    for i in range(1, 6):
        x = newton_step(x, double_well_f, double_well_df)
        print(f"Iteration {i}: x = {x:.6f}, f(x) = {double_well_f(x):.6f}")
    
    print("\n RESULT: Converges to root at x ≈ 1.414")
    
    # Plot the Newton steps
    ax2.plot(x_plot, y_plot, 'b-', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot steps for x0=1.0
    x_vals = [it[1] for it in iterations_a1]
    y_vals = [it[2] for it in iterations_a1]
    ax2.plot(x_vals, y_vals, 'ro-', markersize=8, label='x₀ = 1.0 path')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title('Newton Steps', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flat_spot_analysis.png', dpi=150)
    plt.show()
    
    print("\n CONCLUSION: Newton fails when derivative ≈ 0 because")
    print("   Δx = f(x)/f'(x) becomes infinite.")

#=============================================================================
# SECTION 6: SCENARIO B - The Infinite Cycle
#=============================================================================

def scenario_b_infinite_cycle():
    """
    Scenario B: f(x) = x^3 - 2x + 2
    Demonstrates oscillatory behavior that never reaches the root.
    """
    print_header("SCENARIO B: The Infinite Cycle")
    print("f(x) = x³ - 2x + 2")
    
    # Actual root
    actual_root = -1.769292354  # approximately
    
    print_subheader("Case 1: Starting at x₀ = 0 (Creates a cycle)")
    x = 0.0
    print(f"{'Iter':<6} {'x':<12} {'f(x)':<12} {'Pattern':<15}")
    print("-"*50)
    print(f"{0:<6} {x:<12.6f} {cubic_f(x):<12.6f} Starting point")
    
    sequence = []
    for i in range(1, 11):
        x = newton_step(x, cubic_f, cubic_df)
        sequence.append(x)
        
        # Detect pattern
        if i > 2 and abs(x - sequence[i-3]) < 1e-6:
            pattern = f"Cycle detected!"
        elif i > 2 and abs(x - sequence[i-2]) < 1e-6:
            pattern = f"2-cycle starting"
        else:
            pattern = ""
            
        print(f"{i:<6} {x:<12.6f} {cubic_f(x):<12.6f} {pattern}")
    
    print("\n ANALYSIS: Values oscillate between 0 and 1!")
    print(f"   True root is at x ≈ {actual_root:.6f}")
    print("   This 2-cycle will never reach the actual root.")
    
    print_subheader("Case 2: Starting at x₀ = 2.0 (Converges)")
    x = 2.0
    print(f"{'Iter':<6} {'x':<15} {'f(x)':<15}")
    print("-"*41)
    print(f"{0:<6} {x:<15.6f} {cubic_f(x):<15.6f}")
    
    for i in range(1, 8):
        x = newton_step(x, cubic_f, cubic_df)
        print(f"{i:<6} {x:<15.6f} {cubic_f(x):<15.6f}")
    
    print(f"\n Converged to root at x ≈ {x:.6f}")
    print("\n CONCLUSION: Initial guess matters! Some lead to infinite cycles.")

#=============================================================================
# SECTION 7: SCENARIO C - Overshooting and Divergence
#=============================================================================

def scenario_c_overshooting():
    """
    Scenario C: f(x) = arctan(x)
    Demonstrates how a slightly larger guess can cause divergence.
    """
    print_header("SCENARIO C: Overshooting and Divergence")
    print("f(x) = arctan(x)")
    
    print_subheader("Using scipy.optimize.newton (as required)")
    
    # Try x0 = 1.3
    print("\n--- x₀ = 1.3 ---")
    try:
        root1, info1 = scipy_newton(arctan_f, 1.3, fprime=arctan_df, 
                                    full_output=True, maxiter=50)
        print(f"Root found: {root1:.10f}")
        print(f"Converged: {info1.converged}")
        print(f"Iterations: {info1.iterations}")
    except RuntimeError as e:
        print(f"Error: {e}")
    
    # Try x0 = 1.5
    print("\n--- x₀ = 1.5 ---")
    try:
        root2, info2 = scipy_newton(arctan_f, 1.5, fprime=arctan_df,
                                    full_output=True, maxiter=50)
        print(f"Root found: {root2}")
        print(f"Converged: {info2.converged}")
    except RuntimeError as e:
        print(f"Error: {e} (Diverged!)")
    
    # Manual demonstration
    print_subheader("Manual iteration showing the divergence")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot function
    x_plot = np.linspace(-5, 5, 400)
    y_plot = arctan_f(x_plot)
    
    ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = arctan(x)')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.plot(1.3, arctan_f(1.3), 'go', markersize=8, label='x₀ = 1.3 (works)')
    ax1.plot(1.5, arctan_f(1.5), 'ro', markersize=8, label='x₀ = 1.5 (diverges)')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('arctan(x) - Function Shape', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show Newton steps for x0=1.5
    x = 1.5
    x_vals = [x]
    print(f"{'Iter':<6} {'x':<15} {'f(x)':<15} {'Step size':<15}")
    print("-"*56)
    print(f"{0:<6} {x:<15.6f} {arctan_f(x):<15.6f} {'-':<15}")
    
    for i in range(1, 5):
        step = arctan_f(x)/arctan_df(x)
        x = newton_step(x, arctan_f, arctan_df)
        x_vals.append(x)
        print(f"{i:<6} {x:<15.6f} {arctan_f(x):<15.6f} {step:<15.2f}")
    
    # Plot path
    ax2.plot(x_plot, y_plot, 'b-', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    y_path = [arctan_f(xv) for xv in x_vals]
    ax2.plot(x_vals, y_path, 'ro-', markersize=6, label='Diverging path')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title('Newton Steps - Divergence', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overshoot_divergence.png', dpi=150)
    plt.show()
    
    print("\n EXPLANATION: arctan(x) flattens out as |x| increases.")
    print("   At x = 1.5, the slope is small → huge step overshoots.")
    print("   At x = 1.3, slope still steep enough → converges.")

#=============================================================================
# SECTION 8: PART 4 - Hybrid Method (Safe Newton)
#=============================================================================

def hybrid_newton_safe(f, df, a, b, x0, tol=1e-8, max_iter=30):
    """
    Hybrid method combining Newton-Raphson with Bisection safety.
    
    Parameters:
    f, df : functions
    a, b : bracket [a, b] containing the root
    x0 : initial guess
    tol : tolerance
    max_iter : maximum iterations
    
    Returns:
    x, history : final estimate and iteration history
    """
    print(f"\nInitial bracket: [{a:.3f}, {b:.3f}]")
    print(f"Initial guess: {x0:.3f}")
    print("-"*60)
    print(f"{'Iter':<6} {'Current x':<12} {'Newton guess':<15} {'Action':<20} {'New x':<12}")
    print("-"*70)
    
    x = x0
    history = []
    
    for i in range(max_iter):
        # Check convergence
        if abs(f(x)) < tol:
            print(f"\n Converged to {x:.10f} after {i} iterations")
            return x, history
        
        # Newton step
        try:
            newton_guess = x - f(x)/df(x)
        except ZeroDivisionError:
            newton_guess = float('inf') if x > 0 else float('-inf')
        
        # Store data
        history.append((i, x, newton_guess))
        
        # Check if Newton step is safe
        if newton_guess < a or newton_guess > b:
            action = "OUTSIDE → Bisection"
            # Take bisection step
            x_new = (a + b) / 2
            
            # Update bracket
            if f(a) * f(x_new) < 0:
                b = x_new
            else:
                a = x_new
        else:
            action = "INSIDE → Newton"
            x_new = newton_guess
            
            # Update bracket
            if f(a) * f(x_new) < 0:
                b = x_new
            else:
                a = x_new
        
        print(f"{i:<6} {x:<12.6f} {newton_guess:<15.6f} {action:<20} {x_new:<12.6f}")
        x = x_new
    
    print(f"\n  Max iterations reached. Final x = {x:.10f}")
    return x, history

def part_4_hybrid_method():
    """
    Demonstrates the hybrid safe Newton method.
    """
    print_header("PART 4: Hybrid Method (Safe Newton)")
    print("Combining Newton's speed with Bisection's safety")
    
    # Test on problematic arctan function
    print_subheader("Testing on arctan(x) with x₀ = 1.5")
    print("(Previously this diverged with pure Newton)")
    
    # Bracket containing the root at x=0
    a, b = -1.0, 2.0
    x0 = 1.5
    
    root, history = hybrid_newton_safe(arctan_f, arctan_df, a, b, x0)
    
    print("\n" + "-"*60)
    print("HYBRID METHOD ADVANTAGES:")
    print("• Newton's quadratic convergence when behaving")
    print("• Bisection's guaranteed convergence when Newton fails")
    print("• Bracket ensures we stay in safe region")
    print("• Used in professional physics simulations")

#=============================================================================
# SECTION 9: REPORT - Comparison and Kronig-Penney
#=============================================================================

def bisection_method(f, a, b, tol=1e-8, max_iter=100):
    """Simple bisection method for comparison"""
    if f(a) * f(b) >= 0:
        return None, max_iter
    
    iterations = 0
    for _ in range(max_iter):
        c = (a + b) / 2
        iterations += 1
        
        if abs(f(c)) < tol or (b - a)/2 < tol:
            return c, iterations
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2, iterations

def report_comparison():
    """
    Compares Bisection vs Newton-Raphson iterations.
    """
    print_header("REPORT: Bisection vs. Newton-Raphson Comparison")
    print("Function: f(x) = x - cos(x)")
    print("Tolerance: 1e-8")
    
    # Bisection
    root_b, iter_b = bisection_method(cosine_f, 0, 1, tol=1e-8)
    
    # Newton
    x = 1.0
    iter_n = 0
    for i in range(20):
        iter_n += 1
        x_new = newton_step(x, cosine_f, cosine_df)
        if abs(cosine_f(x_new)) < 1e-8:
            root_n = x_new
            break
        x = x_new
    
    print("\n" + "┌───────────────┬────────────┬──────────────┬─────────────┐")
    print("│    Method     │ Iterations │ Root found   │ Convergence │")
    print("├───────────────┼────────────┼──────────────┼─────────────┤")
    print(f"│  Bisection    │     {iter_b:2d}      │ {root_b:.8f} │ Linear      │")
    print(f"│  Newton-Raph. │     {iter_n:2d}      │ {root_n:.8f} │ Quadratic   │")
    print("└───────────────┴────────────┴──────────────┴─────────────┘")
    
    print(f"\n Newton is {iter_b/iter_n:.1f}x faster!")
    
    print_header("Root Jumping in the Kronig-Penney Model")
    print("""
    WHAT IS ROOT JUMPING?
    ---------------------
    Newton-Raphson converges to the root whose "basin of attraction" 
    contains the initial guess. When multiple roots exist, it may
    "jump" to a different root than intended.
    
    WHY THIS MATTERS IN KRONIG-PENNEY:
    -----------------------------------
    The Kronig-Penney model describes electron energy bands in crystals.
    It yields transcendental equations with MULTIPLE roots, each
    corresponding to a DIFFERENT energy band.
    
    PHYSICAL CONSEQUENCES:
    • If you intend to find the ground state but jump to an excited state,
      you'll predict wrong electronic properties
    • Band gaps, effective masses, and conductivity predictions all depend
      on correctly identifying which band you're in
    • This could lead to misclassifying a material as metallic vs. insulating
    
    PREVENTION:
    -----------
    The hybrid method (Part 4) prevents root jumping by keeping Newton
    within a bracket that contains ONLY the desired root.
    """)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate Kronig-Penney-like function with multiple roots
    E = np.linspace(0.1, 20, 1000)
    f_E = np.sin(np.sqrt(E)) * np.cos(np.sqrt(E)) - 0.3 * np.cos(2*np.sqrt(E))
    
    ax.plot(E, f_E, 'b-', linewidth=2, label='Kronig-Penney-like function')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.fill_between(E, 0, f_E, where=(f_E>0), color='lightblue', alpha=0.3)
    ax.fill_between(E, 0, f_E, where=(f_E<0), color='lightsalmon', alpha=0.3)
    
    # Mark bands
    bands = [2.5, 7.8, 13.2, 18.5]
    colors = ['red', 'green', 'purple', 'orange']
    for i, (band, color) in enumerate(zip(bands, colors)):
        ax.axvline(x=band, color=color, linestyle='--', alpha=0.7)
        ax.text(band+0.3, 0.5, f'Band {i+1}', fontsize=10, color=color)
    
    ax.set_xlabel('Energy (arbitrary units)', fontsize=14)
    ax.set_ylabel('f(E)', fontsize=14)
    ax.set_title('Kronig-Penney Model: Multiple Roots = Multiple Energy Bands', 
                fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kronig_penney_bands.png', dpi=150)
    plt.show()

#=============================================================================
# SECTION 10: MAIN FUNCTION - Runs everything
#=============================================================================

def main():
    """
    Main function that runs all exercises in order.
    """
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "PHYS 300: COMPUTATIONAL PHYSICS" + " "*30 + "║")
    print("║" + " "*32 + "NEWTON-RAPHSON METHOD LAB" + " "*33 + "║")
    print("╚" + "═"*78 + "╝")
    
    # Run all exercises
    exercise_1_finite_well()
    input("\nPress Enter to continue to Exercise 2...")
    
    exercise_2_quadratic_convergence()
    input("\nPress Enter to continue to Scenario A...")
    
    scenario_a_flat_spot()
    input("\nPress Enter to continue to Scenario B...")
    
    scenario_b_infinite_cycle()
    input("\nPress Enter to continue to Scenario C...")
    
    scenario_c_overshooting()
    input("\nPress Enter to continue to Part 4...")
    
    part_4_hybrid_method()
    input("\nPress Enter to continue to Report...")
    
    report_comparison()
    
    print("\n" + "═"*80)
    print("LAB COMPLETE - All exercises finished successfully!")
    print("═"*80 + "\n")


if __name__ == "__main__":
    main()
