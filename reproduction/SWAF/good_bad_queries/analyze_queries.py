"""
    Rapport Comparatif des Requêtes
    --------------------------------
    Requêtes Mauvaises:
    - Total: 48126
    - Contient des <script>: 6698 (13.92%)
    - Contient des SQL: 724 (1.50%)
    - Contient des Commandes Système: 6106 (12.69%)
    - Contient des Caractères Spéciaux: 27301 (56.73%)
    
    Requêtes Bonnes:
    - Total: 1294531
    - Contient des <script>: 0 (0.00%)
    - Contient des SQL: 552 (0.04%)
    - Contient des Commandes Système: 1758 (0.14%)
    - Contient des Caractères Spéciaux: 770 (0.06%)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def read_lines(file_path):
    """
    Lit un fichier ligne par ligne.
    
    Args:
    file_path (str): Chemin vers le fichier à lire.
    
    Returns:
    list: Liste des lignes du fichier.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def analyze_queries(queries):
    """
    Analyse les requêtes pour détecter des patterns communs et des tentatives d'exploitation.
    
    Args:
    queries (list): Liste des requêtes à analyser.
    
    Returns:
    dict: Dictionnaire contenant les statistiques des requêtes.
    """
    stats = {
        'total': len(queries),
        'contains_script': 0,
        'contains_sql': 0,
        'contains_cmd': 0,
        'contains_special_chars': 0,
        'patterns': Counter()
    }
    
    script_pattern = re.compile(r'<script.*?>.*?</script>', re.IGNORECASE)
    sql_pattern = re.compile(r'(select|union|insert|update|delete|drop|truncate|alter|create|replace)\s', re.IGNORECASE)
    cmd_pattern = re.compile(r'(\buname\b|\bwhoami\b|\bpasswd\b|\b/etc/passwd\b|\bid\b)', re.IGNORECASE)
    special_chars_pattern = re.compile(r'[<>\"\'`;()]')
    
    for query in queries:
        if script_pattern.search(query):
            stats['contains_script'] += 1
        if sql_pattern.search(query):
            stats['contains_sql'] += 1
        if cmd_pattern.search(query):
            stats['contains_cmd'] += 1
        if special_chars_pattern.search(query):
            stats['contains_special_chars'] += 1
        stats['patterns'][query.strip()] += 1
    
    return stats

def generate_report(bad_stats, good_stats):
    """
    Génère un rapport comparatif entre les requêtes mauvaises et bonnes.
    
    Args:
    bad_stats (dict): Statistiques des requêtes mauvaises.
    good_stats (dict): Statistiques des requêtes bonnes.
    """
    report = f"""
    Rapport Comparatif des Requêtes
    --------------------------------
    Requêtes Mauvaises:
    - Total: {bad_stats['total']}
    - Contient des <script>: {bad_stats['contains_script']} ({(bad_stats['contains_script']/bad_stats['total'])*100:.2f}%)
    - Contient des SQL: {bad_stats['contains_sql']} ({(bad_stats['contains_sql']/bad_stats['total'])*100:.2f}%)
    - Contient des Commandes Système: {bad_stats['contains_cmd']} ({(bad_stats['contains_cmd']/bad_stats['total'])*100:.2f}%)
    - Contient des Caractères Spéciaux: {bad_stats['contains_special_chars']} ({(bad_stats['contains_special_chars']/bad_stats['total'])*100:.2f}%)
    
    Requêtes Bonnes:
    - Total: {good_stats['total']}
    - Contient des <script>: {good_stats['contains_script']} ({(good_stats['contains_script']/good_stats['total'])*100:.2f}%)
    - Contient des SQL: {good_stats['contains_sql']} ({(good_stats['contains_sql']/good_stats['total'])*100:.2f}%)
    - Contient des Commandes Système: {good_stats['contains_cmd']} ({(good_stats['contains_cmd']/good_stats['total'])*100:.2f}%)
    - Contient des Caractères Spéciaux: {good_stats['contains_special_chars']} ({(good_stats['contains_special_chars']/good_stats['total'])*100:.2f}%)
    """
    
    print(report)
    
    # Générer des graphiques
    labels = ['Scripts', 'SQL', 'Cmds', 'Spéciaux']
    bad_values = [bad_stats['contains_script'], bad_stats['contains_sql'], bad_stats['contains_cmd'], bad_stats['contains_special_chars']]
    good_values = [good_stats['contains_script'], good_stats['contains_sql'], good_stats['contains_cmd'], good_stats['contains_special_chars']]
    
    x = range(len(labels))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, bad_values, width=0.4, label='Mauvaises', align='center')
    plt.bar(x, good_values, width=0.4, label='Bonnes', align='edge')
    plt.xlabel('Types d\'éléments détectés')
    plt.ylabel('Nombre de Requêtes')
    plt.title('Comparaison des Requêtes Mauvaises et Bonnes')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

def main():
    # Lire les fichiers de requêtes
    bad_queries = read_lines('badqueries.txt')
    good_queries = read_lines('goodqueries.txt')
    
    # Analyser les requêtes
    bad_stats = analyze_queries(bad_queries)
    good_stats = analyze_queries(good_queries)
    
    # Générer le rapport
    generate_report(bad_stats, good_stats)

if __name__ == "__main__":
    main()
