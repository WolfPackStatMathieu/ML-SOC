import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import textwrap

def generer_employ_du_temps():
    # Demander la date de création du graphique
    date_str = input("Entrez la date de début (format: YYYY-MM-DD): ")
    date_debut = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Inputs détaillés pour chaque matière
    maths = input("Que souhaitez-vous travailler en mathématiques? ")
    eco = input("Que souhaitez-vous travailler en économie? ")
    socio = input("Que souhaitez-vous travailler en sociologie? ")
    danse = input("Quel aspect de la danse souhaitez-vous travailler cette semaine? ")
    
    # Demande si l'utilisateur souhaite un exemple type ou une personnalisation stricte
    mode_personnalisation = input("Voulez-vous un exemple type d'emploi du temps ou une personnalisation stricte? (exemple/personnalisation) ")

    jours = [(date_debut + timedelta(days=i)).strftime("%A, %d") for i in range(7)]
    
    # Durée de chaque plage horaire en heures
    plage_horaire_durations = {
        'Matin (7h15-9h15)': 2,
        '9h30-11h30': 2,
        '11h30-14h00': 2.5,
        '14h00-16h00': 2,
        '16h00-19h00': 3,
        '19h00-21h00': 2,
        '21h00-23h00': 2,
        '21h00-00h00': 3  # Spécial pour le samedi
    }

    # Répartition du temps pour les matières
    if mode_personnalisation.lower() == 'exemple':
        semaine_type = {
            'Plage horaire': ['Matin (7h15-9h15)', '9h30-11h30', '11h30-14h00', '14h00-16h00', '16h00-19h00', '19h00-21h00', '21h00-23h00'],
            jours[0]: ['Travail', 'Travail', 'Pause déj. + Trajet', 'Travail', 'Travail', 'Activité libre', 'Activité libre'],
            jours[1]: ['Travail', 'Travail', 'Pause déj. + Trajet', 'Travail', 'Travail', 'Activité libre', 'Activité libre'],
            jours[2]: ['Travail', 'Travail', 'Pause déj. + Trajet', 'Travail', 'Travail', 'Activité libre', 'Activité libre'],
            jours[3]: ['Travail', 'Travail', 'Pause déj. + Trajet', 'Travail', 'Travail', 'Activité libre', 'Activité libre'],
            jours[4]: ['Travail', 'Travail', 'Pause déj. + Trajet', 'Travail', 'Travail', 'Activité libre', 'Activité libre'],
            jours[5]: ['Révision socio', 'Révision maths', 'Repos', 'Révision danse (vals)', 'Activité sociale', 'Activité libre', 'Activité libre (jusqu\'à minuit)'],
            jours[6]: ['Révision éco', 'Révision maths', 'Repos', 'Révision danse (vals)', 'Repos', 'Activité libre', 'Activité libre']
        }
        df = pd.DataFrame(semaine_type)
    else:
        # Personnalisation stricte en fonction des proportions et des 37h30 de travail
        print("Vous devez respecter les proportions suivantes pour les révisions : 50% Maths, 25% Sociologie, 25% Économie.")
        total_heures = 37.5  # Heures de travail obligatoires
        heures_travail = 0
        heures_maths = 0
        heures_socio = 0
        heures_eco = 0
        heures_danse = 0

        semaine_type = {'Plage horaire': ['Matin (7h15-9h15)', '9h30-11h30', '11h30-14h00', '14h00-16h00', '16h00-19h00', '19h00-21h00', '21h00-23h00']}

        for jour in jours[:5]:  # Lundi à Vendredi
            activités = []
            heures_jour = 0

            for plage in semaine_type['Plage horaire'][:5]:  # Seulement les heures de travail jusqu'à 19h00
                if heures_travail < 37.5:
                    activité = 'Travail'
                    heures_travail += plage_horaire_durations[plage]
                else:
                    activité = input(f"Que souhaitez-vous faire le {jour} pendant {plage}? ")

                activités.append(activité)

                # Calcul des heures
                duree = plage_horaire_durations[plage]
                heures_jour += duree

                if "maths" in activité:
                    heures_maths += duree
                elif "socio" in activité:
                    heures_socio += duree
                elif "éco" in activité:
                    heures_eco += duree
                elif "danse" in activité:
                    heures_danse += duree

            # Ajout des activités du soir
            for plage in semaine_type['Plage horaire'][5:]:
                activité = input(f"Que souhaitez-vous faire le {jour} pendant {plage}? ")
                activités.append(activité)
                heures_jour += plage_horaire_durations[plage]

            semaine_type[jour] = activités

        # Week-end sans travail
        for jour in jours[5:]:
            activités = []
            for plage in semaine_type['Plage horaire']:
                activité = input(f"Que souhaitez-vous faire le {jour} pendant {plage}? ")
                activités.append(activité)
            semaine_type[jour] = activités

        df = pd.DataFrame(semaine_type)

    # Débogage: Afficher le contenu de la DataFrame
    print("DataFrame créée:")
    print(df)

    # Création du tableau avec des ajustements manuels
    fig, ax = plt.subplots(figsize=(14, 8))  # Ajuster la taille du tableau pour inclure la colonne des jours
    ax.axis('tight')
    ax.axis('off')

    # Titre plus haut sur la figure
    plt.figtext(0.5, 0.95, f'Emploi du Temps Personnalisé à partir du {date_debut.strftime("%A, %d %B")}', fontsize=14, weight='bold', ha='center')

    # Création du tableau
    wrapped_values = [[textwrap.fill(cell, 20) for cell in row] for row in df.values]  # Wrap le texte des cellules
    table = ax.table(cellText=wrapped_values,
                     rowLabels=df['Plage horaire'],
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     edges='closed')

    # Ajuster les hauteurs des lignes et largeurs des colonnes manuellement
    for i, plage_horaire in enumerate(df['Plage horaire']):
        if plage_horaire in plage_horaire_durations:
            idx = df[df['Plage horaire'] == plage_horaire].index
            if not idx.empty:
                idx = idx[0] + 1  # Les indices de la table commencent à 1
                table[(idx, 0)].set_height(plage_horaire_durations[plage_horaire] * 0.02)  # Ajuster la hauteur en proportion
            else:
                print(f"Plage horaire non trouvée dans df: {plage_horaire}")

    # Ajuster les largeurs de colonnes et autres propriétés esthétiques
    for j in range(len(df.columns)):
        table.auto_set_column_width([j])
    
    # Sauvegarder le tableau en image
    plt.savefig('emploi_du_temps.png', bbox_inches='tight', dpi=300)
    print("L'emploi du temps a été sauvegardé sous le nom 'emploi_du_temps.png'.")

    plt.show()

if __name__ == "__main__":
    generer_employ_du_temps()
