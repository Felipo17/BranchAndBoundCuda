import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Konfiguracja stylu wykresów
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def plot_results(data, output_dir):
    # Tworzenie wykresu pudełkowego
    plt.figure()
    sns.boxplot(x="Size", y="TimeMS", hue="Algorithm", data=data)

    plt.yscale("log")
    plt.ylabel("Czas (ms) [skala log]")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'))
    plt.close()


def prepare_speedup_data(df):
    """
    Przetwarza dane: oblicza średnie czasy i Speedup (przyspieszenie)
    względem wersji SequentialOptimized.
    """
    avg_df = df.groupby(['Size', 'Type', 'Algorithm', 'Threads'])['TimeMS'].mean().reset_index()

    base_df = avg_df[avg_df['Algorithm'] == 'SequentialOptimized'][['Size', 'Type', 'TimeMS']]
    base_df = base_df.rename(columns={'TimeMS': 'BaseTime'})

    merged = pd.merge(avg_df, base_df, on=['Size', 'Type'])
    merged['Speedup'] = merged['BaseTime'] / merged['TimeMS']

    merged['Label'] = merged['Algorithm']
    mask_parallel = merged['Algorithm'] == 'Parallel'
    merged.loc[mask_parallel, 'Label'] = merged['Algorithm'] + " (" + merged['Threads'].astype(str) + " thr)"
    
    return merged


def plot_scalability_log(df, output_dir):
    """
    Wykres 1: Skalowalność (Czas vs Rozmiar) w skali logarytmicznej.
    """
    plt.figure()
    sns.lineplot(data=df, x="Size", y="TimeMS", hue="Algorithm", style="Type", markers=True)
    
    plt.yscale("log")
    plt.ylabel("Czas [ms] (log)")
    plt.xlabel("Rozmiar problemu (N)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability.png'))
    plt.close()


def plot_speedup_bar(speedup_df, output_dir):
    """
    Wykres 2: Speedup (Przyspieszenie) względem wersji sekwencyjnej.
    """
    plt.figure()
    plot_data = speedup_df[speedup_df['Algorithm'].isin(['Parallel', 'GPU'])]
    
    sns.barplot(data=plot_data, x="Size", y="Speedup", hue="Label", palette="viridis")
    plt.axhline(1.0, color='red', linestyle='--', label='Brak przyspieszenia (Baseline)')
    
    plt.ylabel("Speedup (x-krotność)")
    plt.xlabel("Rozmiar problemu (N)")
    plt.legend(title="Konfiguracja")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup.png'))
    plt.close()


def plot_heatmap_table(df, output_dir):
    """
    Wykres 4: Heatmapa (tabela kolorów) ze średnimi czasami.
    """
    plt.figure()
    pivot_table = df.pivot_table(index="Size", columns="Algorithm", values="TimeMS", aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Czas [ms]'})
    
    plt.ylabel("Rozmiar (N)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_table.png'))
    plt.close()


def prepare_metrics_data(df):
    """
    Przygotowuje dane do wykresów skalowalności:
    1. Oblicza średni czas (TimeMS) dla każdej konfiguracji.
    2. Wyznacza czas sekwencyjny (Baseline) dla każdego rozmiaru (n).
    3. Oblicza Przyspieszenie (S = T_seq / T_par).
    4. Oblicza Efektywność (E = S / p).
    """
    # 1. Średnie czasy
    avg_df = df.groupby(['Size', 'Type', 'Algorithm', 'Threads'])['TimeMS'].mean().reset_index()

    # 2. Wyodrębnienie czasów sekwencyjnych (Baseline)
    baseline_df = avg_df[avg_df['Algorithm'] == 'SequentialOptimized'][['Size', 'TimeMS']].copy()
    baseline_df.rename(columns={'TimeMS': 'Time_Seq'}, inplace=True)

    # 3. Złączenie danych
    merged_df = pd.merge(avg_df, baseline_df, on='Size', how='inner')

    # 4. Obliczenie S (Speedup)
    merged_df['Speedup'] = merged_df['Time_Seq'] / merged_df['TimeMS']

    # 5. Obliczenie E (Efficiency) = Speedup / Threads
    # Uwaga: Dla GPU (Threads=0) E będzie nieskończoność/NaN, więc filtrujemy to przy wykresach
    # Dla Sequential (Threads=1) E = 1.0
    merged_df['Efficiency'] = merged_df['Speedup'] / merged_df['Threads']
    
    return merged_df


def plot_speedup_vs_n_lines(df, output_dir):
    """
    Wykres liniowy: Wartość S w zależności od n dla różnych p.
    Ignoruje algorytm sekwencyjny i GPU.
    """
    plot_data = df[~df['Algorithm'].isin(['SequentialOptimized', 'GPU'])].copy()
    
    if plot_data.empty:
        print("Brak danych algorytmu równoległego do wygenerowania wykresu S vs n.")
        return

    plt.figure()
    sns.lineplot(
        data=plot_data, 
        x="Size", 
        y="Speedup", 
        hue="Threads", 
        style="Threads",
        palette="viridis", 
        markers='o',
        errorbar=None
    )
    
    plt.xlabel("Rozmiar problemu (n)")
    plt.ylabel("Przyspieszenie (S)")
    plt.legend(title="Liczba procesorów (p)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S_vs_n_by_p.png'))
    plt.close()


def plot_speedup_vs_p_lines(df, output_dir):
    """
    Wykres liniowy: Wartość S w zależności od p dla różnych n.
    """
    plot_data = df[~df['Algorithm'].isin(['SequentialOptimized', 'GPU'])].copy()

    if plot_data.empty:
        return

    plt.figure()
    sns.lineplot(
        data=plot_data, 
        x="Threads", 
        y="Speedup", 
        hue="Size", 
        palette="magma", 
        marker="o",
        errorbar=None
    )
    
    max_p = plot_data['Threads'].max()
    plt.plot([0, max_p], [0, max_p], 'r--', label='Idealne liniowe', alpha=0.5)

    plt.xlabel("Liczba procesorów (p)")
    plt.ylabel("Przyspieszenie (S)")
    plt.legend(title="Rozmiar problemu (n)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S_vs_p_by_n.png'))
    plt.close()


def plot_efficiency_vs_n_lines(df, output_dir):
    """
    Wykres liniowy: Wartość E w zależności od n dla różnych p.
    """
    plot_data = df[~df['Algorithm'].isin(['SequentialOptimized', 'GPU'])].copy()

    if plot_data.empty:
        return

    plt.figure()
    sns.lineplot(
        data=plot_data, 
        x="Size", 
        y="Efficiency", 
        hue="Threads", 
        style="Threads",
        palette="viridis", 
        markers='o',
        errorbar=None
    )
    
    plt.axhline(1.0, color='red', linestyle='--', label='Efektywność idealna', alpha=0.7)

    plt.xlabel("Rozmiar problemu (n)")
    plt.ylabel("Efektywność (E)")
    plt.legend(title="Liczba procesorów (p)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'E_vs_n_by_p.png'))
    plt.close()


def plot_efficiency_vs_p_lines(df, output_dir):
    """
    Wykres liniowy: Wartość E w zależności od p dla różnych n.
    """
    plot_data = df[~df['Algorithm'].isin(['SequentialOptimized', 'GPU'])].copy()

    if plot_data.empty:
        return

    plt.figure()
    sns.lineplot(
        data=plot_data, 
        x="Threads", 
        y="Efficiency", 
        hue="Size", 
        palette="magma", 
        marker="o",
        errorbar=None
    )

    plt.axhline(1.0, color='red', linestyle='--', label='Efektywność idealna', alpha=0.7)
    
    plt.xlabel("Liczba procesorów (p)")
    plt.ylabel("Efektywność (E)")
    plt.legend(title="Rozmiar problemu (n)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'E_vs_p_by_n.png'))
    plt.close()


if __name__ == "__main__":
    file_path = "data\\results.csv" 
    output_dir = "plots"

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    
    plot_results(df, output_dir)
    plot_scalability_log(df, output_dir)
    plot_speedup_bar(prepare_speedup_data(df), output_dir)
    plot_heatmap_table(df, output_dir)

    # 1. Przetworzenie danych (S i E)
    metrics_df = prepare_metrics_data(df)
    # print(metrics_df)

    # 2. Generowanie nowych wykresów
    plot_speedup_vs_n_lines(metrics_df, output_dir)
    plot_speedup_vs_p_lines(metrics_df, output_dir)
    plot_efficiency_vs_n_lines(metrics_df, output_dir)
    plot_efficiency_vs_p_lines(metrics_df, output_dir)
