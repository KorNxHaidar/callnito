import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    if not os.path.exists("eval_results.csv"):
        print("Error: eval_results.csv not found.")
        return

    try:
        df = pd.read_csv("eval_results.csv")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV is empty.")
        return

    # Calculate overall accuracy
    acc_no_rag = df['correct_no_rag'].mean()
    acc_rag = df['correct_rag'].mean()

    print(f"Overall Accuracy (No RAG): {acc_no_rag:.2%}")
    print(f"Overall Accuracy (With RAG): {acc_rag:.2%}")

    # Prepare data for plotting
    # We want to compare Overall, and maybe breakdown by Ground Truth if available
    
    plot_data = []
    
    # Overall
    plot_data.append({'Method': 'No RAG', 'Category': 'Overall', 'Accuracy': acc_no_rag * 100})
    plot_data.append({'Method': 'With RAG', 'Category': 'Overall', 'Accuracy': acc_rag * 100})

    # By Ground Truth Type
    if 'ground_truth' in df.columns:
        # Scam (1)
        scam_df = df[df['ground_truth'] == 1]
        if not scam_df.empty:
            acc_no_rag_scam = scam_df['correct_no_rag'].mean()
            acc_rag_scam = scam_df['correct_rag'].mean()
            plot_data.append({'Method': 'No RAG', 'Category': 'Scam (Positive)', 'Accuracy': acc_no_rag_scam * 100})
            plot_data.append({'Method': 'With RAG', 'Category': 'Scam (Positive)', 'Accuracy': acc_rag_scam * 100})
        
        # Safe (0)
        safe_df = df[df['ground_truth'] == 0]
        if not safe_df.empty:
            acc_no_rag_safe = safe_df['correct_no_rag'].mean()
            acc_rag_safe = safe_df['correct_rag'].mean()
            plot_data.append({'Method': 'No RAG', 'Category': 'Safe (Negative)', 'Accuracy': acc_no_rag_safe * 100})
            plot_data.append({'Method': 'With RAG', 'Category': 'Safe (Negative)', 'Accuracy': acc_rag_safe * 100})

    plot_df = pd.DataFrame(plot_data)

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Use a nice seaborn theme
    sns.set_theme(style="white")
    
    # Bar plot
    hue_order = ['No RAG', 'With RAG']
    palette = {'No RAG': '#89CFF0', 'With RAG': '#0047AB'}
    
    ax = sns.barplot(
        data=plot_df, 
        x='Category', 
        y='Accuracy', 
        hue='Method',
        palette=palette,
        hue_order=hue_order
    )

    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)

    # Customize axes
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.title("Scam Detection Accuracy Evaluation")
    plt.legend(loc='lower right')
    sns.despine(left=True, bottom=True)

    # Save
    output_file = "eval_plot.jpg"
    plt.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
