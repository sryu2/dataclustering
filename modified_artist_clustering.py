import csv
from gurobipy import Model, GRB, quicksum

# Step 1: Load the Data from a Local CSV File
def load_data(file_path):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numeric fields to appropriate types
            for key in row:
                try:
                    row[key] = float(row[key]) if row[key].replace('.', '', 1).isdigit() else row[key]
                except ValueError:
                    row[key] = 0  # Replace invalid entries with 0
            data.append(row)
    return data

# Step 2: Define the Ideal Profiles and Weights
profiles = {
    'Ready': {
        'profile': {
            'Number of Songs (Spotify)': 3,  # At least 3 songs
            'Monthly listeners (Spotify)': 5000,  # Greater than 5,000
            'Total Streams (Spotify)': 10000,  # Greater than 10,000
            'Fan Retention Rate (Spotify)': 10,  # Greater than 10%
            'Playlist Reach (Spotify)': 10000,  # Greater than 10,000
            'Platform Playlists appearence (Spotify)': 1,  # At least 1 playlist appearance
            'Non-platform playlists (Spotify)': 50,  # More than 50
            'Spotify Following': 1000,  # More than 1,000
            'Instagram Following': 1000,  # More than 1,000
            'TikTok Following': 10000  # More than 10,000
        },
        'weight': 1
    },
    'Potential': {
        'profile': {
            'Number of Songs (Spotify)': 2,  # At least 2 songs
            'Monthly listeners (Spotify)': 2000,  # Greater than 2,000
            'Total Streams (Spotify)': 5000,  # Greater than 5,000
            'Fan Retention Rate (Spotify)': 5,  # Greater than 5%
            'Playlist Reach (Spotify)': 5000,  # Greater than 5,000
            'Platform Playlists appearence (Spotify)': 0,  # At least 0 playlist appearance
            'Non-platform playlists (Spotify)': 20,  # More than 20
            'Spotify Following': 500,  # More than 500
            'Instagram Following': 500,  # More than 500
            'TikTok Following': 5000  # More than 5,000
        },
        'weight': 0.5
    },
    'Not Ready': {
        'profile': {
            'Number of Songs (Spotify)': 0,
            'Monthly listeners (Spotify)': 0,
            'Total Streams (Spotify)': 0,
            'Fan Retention Rate (Spotify)': 0,
            'Playlist Reach (Spotify)': 0,
            'Platform Playlists appearence (Spotify)': 0,
            'Non-platform playlists (Spotify)': 0,
            'Spotify Following': 0,
            'Instagram Following': 0,
            'TikTok Following': 0
        },
        'weight': 0
    }
}

# Step 3: Calculate Weighted Manhattan Distance

def calculate_distance(row, profile):
    distance = 0
    for feature, ideal_value in profile.items():
        try:
            row_value = row.get(feature, 0)  # Use 0 if feature is missing
            distance += abs(float(row_value) - ideal_value)
        except ValueError:
            print(f"Non-numeric value for feature '{feature}' in row: {row}")
            distance += ideal_value  # Assign max possible penalty for invalid data
        except Exception as e:
            print(f"Error processing feature '{feature}': {e}")
            raise
    return distance

def normalize_distances(data, profiles):
    for cluster_name in profiles.keys():
        distances = [row[f'Distance_to_{cluster_name}'] for row in data]
        max_distance = max(distances) if distances else 1
        for row in data:
            row[f'Distance_to_{cluster_name}'] /= max_distance


def calculate_all_distances(data, profiles):
    for row in data:
        for cluster_name, cluster_data in profiles.items():
            profile = cluster_data['profile']
            row[f'Distance_to_{cluster_name}'] = calculate_distance(row, profile)
    normalize_distances(data, profiles)
    return data

# Step 4: Formulate and Solve the Optimization Model
def cluster_artists(data, profiles):
    model = Model("Artist Clustering")

    # Add decision variables for each cluster
    x = model.addVars(len(data), len(profiles), vtype=GRB.BINARY, name="x")

    # Objective: Minimize the total normalized distance for all clusters
    model.setObjective(
        quicksum(
            x[i, j] * (data[i][f'Distance_to_{cluster_name}'] + (10 if cluster_name == "Not Ready" else 0))
            for i in range(len(data))
            for j, cluster_name in enumerate(profiles.keys())
        ),
        GRB.MINIMIZE
    )

    # Constraint: Each artist must belong to exactly one cluster
    for i in range(len(data)):
        model.addConstr(quicksum(x[i, j] for j in range(len(profiles))) == 1, f"Artist_{i}_Cluster_Assignment")

    # Minimum number of artists in each cluster
    min_artists = max(1, len(data) // len(profiles))  # Ensure at least one artist per cluster
    for j, cluster_name in enumerate(profiles.keys()):
        model.addConstr(quicksum(x[i, j] for i in range(len(data))) >= min_artists, f"Min_{cluster_name}")

    # Optimize the model
    model.optimize()

    # Assign clusters based on the optimization result
    cluster_names = list(profiles.keys())
    for i in range(len(data)):
        for j, cluster_name in enumerate(cluster_names):
            if x[i, j].X > 0.5:
                data[i]['Cluster'] = cluster_name
                break

    return data

# Step 5: Main Function to Run the Entire Workflow
def main():
    # Path to your local CSV file
    file_path = 'artist_data.csv'

    print("Loading data...")
    artist_data = load_data(file_path)

    print("Calculating distances to the ideal profiles...")
    artist_data = calculate_all_distances(artist_data, profiles)

    print("Clustering artists...")
    clustered_data = cluster_artists(artist_data, profiles)

    print("Clustering complete. Displaying results:")
    for artist in clustered_data:
        print(f"Artist: {artist.get('Artist Name', 'Unknown')}, Cluster: {artist['Cluster']}")

    # Save results to a new CSV file
    output_file = 'modified_clustered_artists.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=clustered_data[0].keys())
        writer.writeheader()
        writer.writerows(clustered_data)

    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
