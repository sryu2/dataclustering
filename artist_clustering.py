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

# Step 2: Define the Ideal Artist Profile and Weights
ideal_artist = {
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
}

weights = {
    'Number of Songs (Spotify)': 1,
    'Monthly listeners (Spotify)': 2,
    'Total Streams (Spotify)': 2,
    'Fan Retention Rate (Spotify)': 1.5,
    'Playlist Reach (Spotify)': 1.5,
    'Platform Playlists appearence (Spotify)': 1,
    'Non-platform playlists (Spotify)': 1,
    'Spotify Following': 1,
    'Instagram Following': 1,
    'TikTok Following': 1
}

# Step 3: Calculate Weighted Manhattan Distance
def calculate_distance(row, ideal, weights):
    distance = 0
    for feature, ideal_value in ideal.items():
        weight = weights.get(feature, 1)
        try:
            row_value = row.get(feature, 0)  # Use 0 if feature is missing
            distance += weight * abs(float(row_value) - ideal_value)
        except ValueError:
            print(f"Non-numeric value for feature '{feature}' in row: {row}")
            distance += weight * ideal_value  # Assign max possible penalty for invalid data
        except Exception as e:
            print(f"Error processing feature '{feature}': {e}")
            raise
    return distance

def calculate_all_distances(data, ideal, weights):
    for row in data:
        row['Distance_to_Ideal'] = calculate_distance(row, ideal, weights)
    return data

# Step 4: Formulate and Solve the Optimization Model
def cluster_artists(data, min_ready_artists=5):
    model = Model("Artist Clustering")

    # Add decision variables: x[i] = 1 if artist i is in the 'Ready' cluster, 0 otherwise
    x = model.addVars(len(data), vtype=GRB.BINARY, name="x")

    # Objective: Minimize the total weighted distance for the 'Ready' cluster
    model.setObjective(
        quicksum(x[i] * data[i]['Distance_to_Ideal'] for i in range(len(data))), GRB.MINIMIZE
    )

    # Constraint: At least `min_ready_artists` must be in the 'Ready' cluster
    model.addConstr(quicksum(x[i] for i in range(len(data))) >= min_ready_artists, "min_ready_artists")

    # Optimize the model
    model.optimize()

    # Assign clusters based on the optimization result
    for i in range(len(data)):
        data[i]['Cluster'] = 'Ready' if x[i].X > 0.5 else 'Not Ready'

    return data

# Step 5: Main Function to Run the Entire Workflow
def main():
    # Path to your local CSV file
    file_path = 'artist_data.csv'

    print("Loading data...")
    artist_data = load_data(file_path)

    print("Calculating distances to the ideal profile...")
    artist_data = calculate_all_distances(artist_data, ideal_artist, weights)

    print("Clustering artists...")
    clustered_data = cluster_artists(artist_data, min_ready_artists=5)

    print("Clustering complete. Displaying results:")
    for artist in clustered_data:
        print(f"Artist: {artist.get('Artist Name', 'Unknown')}, Distance: {artist['Distance_to_Ideal']}, Cluster: {artist['Cluster']}")

    # Save results to a new CSV file
    output_file = 'clustered_artists.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=clustered_data[0].keys())
        writer.writeheader()
        writer.writerows(clustered_data)

    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
