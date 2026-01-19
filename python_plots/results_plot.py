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
    # POPRAWKA: użycie 'data' zamiast globalnego 'df'
    sns.boxplot(x="Size", y="TimeMS", hue="Algorithm", data=data)

    # POPRAWKA: plt.title zamiast plt.set_title (to samo dla yscale i ylabel)
    plt.title("Porównanie czasów wykonania algorytmów")
    plt.yscale("log")
    plt.ylabel("Czas (ms) [skala log]")
    
    plt.tight_layout()
    # POPRAWKA: bezpieczne łączenie ścieżek
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
    
    # POPRAWKI metod plt
    plt.yscale("log")
    plt.title("Skalowalność: Czas wykonania vs Rozmiar (Skala Log)")
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
    
    # POPRAWKI metod plt
    plt.title("Przyspieszenie względem SequentialOptimized")
    plt.ylabel("Speedup (x-krotność)")
    plt.xlabel("Rozmiar problemu (N)")
    plt.legend(title="Konfiguracja")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup.png'))
    plt.close()


def plot_correlation_impact(df, output_dir):
    """
    Wykres 3: Wpływ korelacji danych na czas wykonania (Boxplot).
    """
    plt.figure()
    sns.boxplot(data=df, x="Type", y="TimeMS", hue="Algorithm")
    
    # POPRAWKI metod plt
    plt.yscale("log")
    plt.title("Rozkład czasów zależnie od korelacji danych")
    plt.ylabel("Czas [ms] (log)")
    plt.xlabel("Typ danych")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_impact.png'))
    plt.close()


def plot_heatmap_table(df, output_dir):
    """
    Wykres 4: Heatmapa (tabela kolorów) ze średnimi czasami.
    """
    plt.figure()
    pivot_table = df.pivot_table(index="Size", columns="Algorithm", values="TimeMS", aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Czas [ms]'})
    
    # POPRAWKI metod plt
    plt.title("Mapa ciepła: Średnie czasy wykonania [ms]")
    plt.ylabel("Rozmiar (N)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_table.png'))
    plt.close()


if __name__ == "__main__":
    file_path = "data\\results.csv" 
    output_dir = "plots"

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    
    plot_results(df, output_dir)
    plot_scalability_log(df, output_dir)
    plot_speedup_bar(prepare_speedup_data(df), output_dir)
    plot_correlation_impact(df, output_dir)
    plot_heatmap_table(df, output_dir)
