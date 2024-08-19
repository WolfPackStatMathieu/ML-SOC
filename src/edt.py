import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import textwrap

# Dictionnaire pour traduire les jours en français
jours_francais = {
    'Monday': 'Lundi',
    'Tuesday': 'Mardi',
    'Wednesday': 'Mercredi',
    'Thursday': 'Jeudi',
    'Friday': 'Vendredi',
    'Saturday': 'Samedi',
    'Sunday': 'Dimanche'
}

def generer_employ_du_temps():
    # Demander la date de création du graphique
    date_str = input("Entrez la date de début (format: YYYY-MM-DD): ")
    date_debut = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Inputs détaillés pour chaque matière
    maths = input("Que souhaitez-vous travailler en mathématiques? (écrire 'maths') ")
    eco = input("Que souhaitez-vous travailler en économie? (écrire 'éco') ")
    socio = input("Que souhaitez-vous travailler en sociologie? (écrire 'socio') ")
    danse = input("Quel aspect de la danse souhaitez-vous travailler cette semaine? (écrire 'danse') ")
    
    # Demande si l'utilisateur souhaite un exemple type, une personnalisation stricte, ou une génération automatique
    mode_personnalisation = input("Voulez-vous un exemple type d'emploi du temps, une personnalisation stricte, ou une génération automatique? (exemple/personnalisation/auto) ")

    # Obtenir les jours et les traduire en français
    jours = [(date_debut + timedelta(days=i)).strftime("%A, %d") for i in range(7)]
    jours = [jours_francais[day.split(',')[0]] + day[day.index(','):] for day in jours]
    
    # Définir la durée de chaque plage horaire en heures
    plage_horaire_durations = {
        'Matin (7h15-9h15)': 2,
        '9h30-11h30': 2,  # Travail obligatoire en semaine
        '11h30-14h00': 2.5,
        '14h00-16h00': 2,  # Travail obligatoire en semaine
        '16h00-19h00': 3,
        '19h00-21h00': 2,
        '21h00-23h00': 2,
        '21h00-00h00': 3  # Spécial pour le samedi
    }

    # Initialiser les compteurs d'heures
    heures_danse = 0
    heures_maths = 0
    heures_socio = 0
    heures_eco = 0
    heures_travail = 0

    # Répartition du temps pour les matières
    semaine_type = {'Plage horaire': ['Matin (7h15-9h15)', '9h30-11h30', '11h30-14h00', '14h00-16h00', '16h00-19h00', '19h00-21h00', '21h00-23h00']}

    if mode_personnalisation.lower() == 'exemple':
        total_heures_personnalisables = 45.0
        heures_danse = total_heures_personnalisables * 0.5
        heures_concours = total_heures_personnalisables * 0.5
        heures_maths = heures_concours * 0.5
        heures_eco = heures_concours * 0.25
        heures_socio = heures_concours * 0.25

        for jour in jours[:5]:  # Lundi à Vendredi
            semaine_type[jour] = [
                'Travail',            # Matin (7h15-9h15)
                'Travail',            # 9h30-11h30 (obligatoire)
                'Pause déj.',         # 11h30-14h00
                'Travail',            # 14h00-16h00 (obligatoire)
                'Travail',            # 16h00-19h00
                'Activité libre',     # 19h00-21h00
                'Activité libre'      # 21h00-23h00
            ]
            heures_travail += plage_horaire_durations['Matin (7h15-9h15)'] + plage_horaire_durations['9h30-11h30'] + plage_horaire_durations['14h00-16h00'] + plage_horaire_durations['16h00-19h00']

        for jour in jours[5:]:  # Samedi et Dimanche
            semaine_type[jour] = [
                'Révision socio',     # Matin (7h15-9h15)
                'Révision maths',     # 9h30-11h30
                'Repos',              # 11h30-14h00
                'Révision danse (vals)', # 14h00-16h00
                'Activité sociale',   # 16h00-19h00
                'Activité libre',     # 19h00-21h00
                'Activité libre' if jour != jours[5] else 'Activité libre (jusqu\'à minuit)'  # 21h00-23h00 ou 21h00-00h00 le samedi
            ]

        df = pd.DataFrame(semaine_type)
    
    elif mode_personnalisation.lower() == 'auto':
        # Génération automatique avec respect des contraintes
        heures_travail_cible = float(input("Combien d'heures de travail souhaitez-vous programmer cette semaine? (entre 37,5 et 45 heures) "))
        if heures_travail_cible < 37.5 or heures_travail_cible > 45:
            print("Le nombre d'heures de travail doit être compris entre 37,5 et 45.")
            return
        
        total_heures_personnalisables = 45.0
        heures_concours = total_heures_personnalisables * 0.5
        heures_danse_cible = total_heures_personnalisables * 0.5
        heures_maths_cible = heures_concours * 0.5
        heures_eco_cible = heures_concours * 0.25
        heures_socio_cible = heures_concours * 0.25

        for jour in jours[:5]:  # Lundi à Vendredi
            activités = []
            heures_jour = 0

            # Plages horaires obligatoires pour le travail
            activités.append('Travail')  # 9h30-11h30
            activités.append('Travail')  # 14h00-16h00
            heures_jour += plage_horaire_durations['9h30-11h30'] + plage_horaire_durations['14h00-16h00']

            # Ajout des créneaux supplémentaires si nécessaire
            if heures_travail + heures_jour < heures_travail_cible:
                activités.insert(0, 'Travail')  # Matin (7h15-9h15)
                heures_jour += plage_horaire_durations['Matin (7h15-9h15)']

            if heures_travail + heures_jour < heures_travail_cible:
                activités.append('Travail')  # 16h00-19h00
                heures_jour += plage_horaire_durations['16h00-19h00']

            heures_travail += heures_jour
            semaine_type[jour] = activités + ['Pause déj.', 'Activité libre', 'Activité libre']

        # Remplissage pour le week-end
        for jour in jours[5:]:
            activités = [
                'Révision socio' if heures_socio < heures_socio_cible else 'Révision maths',
                'Révision maths' if heures_maths < heures_maths_cible else 'Révision éco',
                'Repos',
                'Révision danse (vals)',
                'Activité sociale',
                'Activité libre',
                'Activité libre' if jour != jours[5] else 'Activité libre (jusqu\'à minuit)'
            ]

            # Mise à jour des heures
            heures_socio += plage_horaire_durations['Matin (7h15-9h15)'] if heures_socio < heures_socio_cible else 0
            heures_maths += plage_horaire_durations['9h30-11h30'] if heures_maths < heures_maths_cible else 0
            heures_eco += plage_horaire_durations['14h00-16h00'] if heures_eco < heures_eco_cible else 0
            heures_danse += plage_horaire_durations['16h00-19h00']

            semaine_type[jour] = activités
        
        # Ajustement pour uniformiser la longueur des listes de chaque jour
        for jour in jours:
            if len(semaine_type[jour]) < len(semaine_type['Plage horaire']):
                semaine_type[jour] += [''] * (len(semaine_type['Plage horaire']) - len(semaine_type[jour]))

        df = pd.DataFrame(semaine_type)
    
    else:
        # Personnalisation stricte en fonction des proportions
        total_heures_personnalisables = 45.0
        heures_danse_cible = total_heures_personnalisables * 0.5
        heures_concours_cible = total_heures_personnalisables * 0.5
        heures_maths_cible = heures_concours_cible * 0.5
        heures_eco_cible = heures_concours_cible * 0.25
        heures_socio_cible = heures_concours_cible * 0.25

        print(f"Temps total disponible : {total_heures_personnalisables:.1f} heures")
        print(f"Vous devez répartir ce temps de manière égale entre la danse et le concours.")
        print(f"Pour le concours : {heures_maths_cible:.1f} heures pour les maths, {heures_eco_cible:.1f} heures pour l'économie, et {heures_socio_cible:.1f} heures pour la sociologie.")

        for jour in jours[:5]:  # Lundi à Vendredi
            activités = []
            heures_jour = 0

            # Plages horaires obligatoires pour le travail
            activités.append('Travail')  # 9h30-11h30 (obligatoire)
            activités.append('Travail')  # 14h00-16h00 (obligatoire)
            heures_jour += plage_horaire_durations['9h30-11h30'] + plage_horaire_durations['14h00-16h00']

            # Ajout des activités du soir
            for plage in semaine_type['Plage horaire'][5:]:
                print(f"Il vous reste {heures_danse_cible - heures_danse:.1f} heures à allouer à la danse.")
                print(f"Il vous reste {heures_maths_cible - heures_maths:.1f} heures à allouer aux maths.")
                print(f"Il vous reste {heures_eco_cible - heures_eco:.1f} heures à allouer à l'économie.")
                print(f"Il vous reste {heures_socio_cible - heures_socio:.1f} heures à allouer à la sociologie.")
                activité = input(f"Que souhaitez-vous faire le {jour} pendant {plage}? (pour maths écrire 'maths', pour sociologie écrire 'socio', pour économie écrire 'éco', pour danse écrire 'danse', pour travail écrire 'travail') ")
                activités.append(activité)
                if "maths" in activité:
                    heures_maths += plage_horaire_durations[plage]
                elif "socio" in activité:
                    heures_socio += plage_horaire_durations[plage]
                elif "éco" in activité:
                    heures_eco += plage_horaire_durations[plage]
                elif "danse" in activité:
                    heures_danse += plage_horaire_durations[plage]
                elif "travail" in activité:
                    if heures_travail < 37.5 and plage not in ["19h00-21h00", "21h00-23h00", "21h00-00h00"]:
                        heures_travail += plage_horaire_durations[plage]
                
            heures_travail += heures_jour
            semaine_type[jour] = activités

        # Week-end sans travail obligatoire
        for jour in jours[5:]:
            activités = []
            for plage in semaine_type['Plage horaire']:
                print(f"Il vous reste {heures_danse_cible - heures_danse:.1f} heures à allouer à la danse.")
                print(f"Il vous reste {heures_maths_cible - heures_maths:.1f} heures à allouer aux maths.")
                print(f"Il vous reste {heures_eco_cible - heures_eco:.1f} heures à allouer à l'économie.")
                print(f"Il vous reste {heures_socio_cible - heures_socio:.1f} heures à allouer à la sociologie.")
                activité = input(f"Que souhaitez-vous faire le {jour} pendant {plage}? (pour maths écrire 'maths', pour sociologie écrire 'socio', pour économie écrire 'éco', pour danse écrire 'danse') ")
                activités.append(activité)
                if "maths" in activité:
                    heures_maths += plage_horaire_durations[plage]
                elif "socio" in activité:
                    heures_socio += plage_horaire_durations[plage]
                elif "éco" in activité:
                    heures_eco += plage_horaire_durations[plage]
                elif "danse" in activité:
                    heures_danse += plage_horaire_durations[plage]
                
            semaine_type[jour] = activités
        
        # Ajustement pour uniformiser la longueur des listes de chaque jour
        for jour in jours:
            if len(semaine_type[jour]) < len(semaine_type['Plage horaire']):
                semaine_type[jour] += [''] * (len(semaine_type['Plage horaire']) - len(semaine_type[jour]))

        df = pd.DataFrame(semaine_type)

    # Calcul des temps totaux pour la danse, le concours, et le travail
    total_danse = heures_danse
    total_concours = heures_maths + heures_socio + heures_eco
    total_travail = heures_travail

    # Vérification du respect du minimum de 37,5 heures de travail
    if total_travail < 37.5:
        print(f"Attention : Vous n'avez alloué que {total_travail:.1f} heures de travail. Il est nécessaire d'atteindre au moins 37,5 heures de travail.")

    # Ajustement des hauteurs des cellules différenciées
    cell_heights = {
        'Matin (7h15-9h15)': 0.15,
        '9h30-11h30': 0.15,
        '11h30-14h00': 0.2,
        '14h00-16h00': 0.15,
        '16h00-19h00': 0.2,
        '19h00-21h00': 0.15,
        '21h00-23h00': 0.15,
        '21h00-00h00': 0.2  # Spécial pour le samedi
    }

    # Amélioration de l'aspect visuel du tableau avec grille et alternance de couleurs
    fig, ax = plt.subplots(figsize=(8.3, 5.8))  # Taille du tableau pour une demi-feuille A4
    ax.axis('tight')
    ax.axis('off')

    # Titre plus haut sur la figure
    plt.figtext(0.5, 0.98, f'Emploi du Temps Personnalisé à partir du {date_debut.strftime("%A, %d %B")}', fontsize=14, weight='bold', ha='center')

    # Création du tableau
    wrapped_values = [[textwrap.fill(cell, 20) for cell in row] for row in df.iloc[:, 1:].values]  # Wrap le texte des cellules
    table = ax.table(cellText=wrapped_values,
                     rowLabels=df['Plage horaire'],
                     colLabels=jours,
                     cellLoc='center',
                     loc='center',
                     edges='closed')

    # Ajustement des propriétés des cellules pour une meilleure lisibilité
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.0)

    # Appliquer la grille et alterner les couleurs de fond pour les cellules
    for (i, j), cell in table.get_celld().items():
        if j == -1:
            cell.set_fontsize(10)
            cell.set_text_props(weight='bold')
            if i > 0:
                cell.set_height(cell_heights.get(df['Plage horaire'].iloc[i-1], 0.15))
        elif i == 0:
            cell.set_fontsize(10)
            cell.set_text_props(weight='bold')
        else:
            cell.set_height(cell_heights.get(df['Plage horaire'].iloc[i-1], 0.15))
            cell.set_text_props(ha='center', va='center', wrap=True)
            cell.set_linewidth(1.5)
            # Alterner les couleurs
            if (i + j) % 2 == 0:
                cell.set_facecolor('#f0f0f0')  # Gris clair
            else:
                cell.set_facecolor('white')

    # Afficher les temps totaux en bas de la figure, en dessous du tableau
    plt.figtext(0.5, -0.1, f"Temps total consacré au travail : {total_travail:.1f} heures\n"
                           f"Temps total consacré à la danse : {total_danse:.1f} heures\n"
                           f"Temps total consacré au concours : {total_concours:.1f} heures",
                fontsize=10, ha='center', weight='bold')

    # Enregistrer le tableau dans un fichier image
    plt.savefig('emploi_du_temps_final.png', bbox_inches='tight')
    print("Le tableau a été enregistré sous 'emploi_du_temps_final.png'.")

# Exécuter la fonction
generer_employ_du_temps()
