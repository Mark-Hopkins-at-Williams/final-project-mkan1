import matplotlib.pyplot as plt

def plot_gsm8k():
    # Data
    sizes = [1, 2, 3]
    no_cot_accuracies = [3.5, 2.79, 12.6]
    cot_accuracies = [3.4, 4.79, 36.2]

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(sizes, no_cot_accuracies, color='blue', label='No CoT')
    plt.scatter(sizes, cot_accuracies, color='red', label='CoT')

    # Plot trend lines
    plt.plot(sizes, no_cot_accuracies, color='blue', linestyle='-')
    plt.plot(sizes, cot_accuracies, color='red', linestyle='-')

    # Ticks
    plt.xticks([1, 2, 3], labels=[str("1"), str("3"), str("175")], )


    # Labels and title
    plt.xlabel('Model Scale (# parameters in billions)')
    plt.ylabel('Solve Rate (%)')
    plt.title('Model Accuracy vs Size on GSM8K')
    plt.legend()

    plt.show()

def plot_stratqa():
    # Data
    sizes = [1, 2, 3]
    no_cot_accuracies = [48.8, 56.5, 59.4]
    cot_accuracies = [56.5, 54.8, 70.1]

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(sizes, no_cot_accuracies, color='blue', label='No CoT')
    plt.scatter(sizes, cot_accuracies, color='red', label='CoT')

    # Plot trend lines
    plt.plot(sizes, no_cot_accuracies, color='blue', linestyle='-')
    plt.plot(sizes, cot_accuracies, color='red', linestyle='-')

    # Ticks
    plt.xticks([1, 2, 3], labels=[str("1"), str("3"), str("175")], )


    # Labels and title
    plt.xlabel('Model Scale (# parameters in billions)')
    plt.ylabel('Solve Rate (%)')
    plt.title('Model Accuracy vs Size on StrategyQA')
    plt.legend()

    plt.show()

def plot_in_domain():
    # Data
    sizes = [1, 2, 3]
    no_cot_accuracies = [0.6, 2.4, 1.0]
    cot_accuracies = [3.2, 6.2, 93.2]

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(sizes, no_cot_accuracies, color='blue', label='No CoT')
    plt.scatter(sizes, cot_accuracies, color='red', label='CoT')

    # Plot trend lines
    plt.plot(sizes, no_cot_accuracies, color='blue', linestyle='-')
    plt.plot(sizes, cot_accuracies, color='red', linestyle='-')

    # Ticks
    plt.xticks([1, 2, 3], labels=[str("1"), str("3"), str("175")], )


    # Labels and title
    plt.xlabel('Model Scale (# parameters in billions)')
    plt.ylabel('Solve Rate (%)')
    plt.title('Model Accuracy vs Size on Last Letter Concatenation In Domain')
    plt.legend()

    plt.show()

def plot_out_domain():
    # Data
    sizes = [1, 2, 3]
    no_cot_accuracies = [0.0, 0.0, 0]
    cot_accuracies = [0.0, 0.2, 30.2]

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(sizes, no_cot_accuracies, color='blue', label='No CoT')
    plt.scatter(sizes, cot_accuracies, color='red', label='CoT')

    # Plot trend lines
    plt.plot(sizes, no_cot_accuracies, color='blue', linestyle='-')
    plt.plot(sizes, cot_accuracies, color='red', linestyle='-')

    # Ticks
    plt.xticks([1, 2, 3], labels=[str("1"), str("3"), str("175")], )


    # Labels and title
    plt.xlabel('Model Scale (# parameters in billions)')
    plt.ylabel('Solve Rate (%)')
    plt.title('Model Accuracy vs Size on Last Letter Concatenation Out of Domain')
    plt.legend()

    plt.show()