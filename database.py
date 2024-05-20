import firebase_admin
from firebase_admin import credentials


from firebase_admin import firestore

cred = credentials.Certificate("python/key/app-be149-firebase-adminsdk-q5m3j-38fa27ada9.json")
firebase_admin.initialize_app(cred)

class CarSearch:
    def __init__(self):
        # Initialize Firestore database
        self.db = firestore.client()
        
    @staticmethod
    def english_to_arabic(english_letters):
        mapping = {
            'A': 'أ',
            'B': 'ب',
            'J': 'ح',
            'D': 'د',
            'R': 'ر',
            'S': 'س',
            'X': 'ص',
            'T': 'ط',
            'E': 'ع',
            'G': 'ق',
            'K': 'ك',
            'L': 'ل',
            'Z': 'م',
            'N': 'ن',
            'H': 'هـ',
            'U': 'و',
            'V': 'ى',
        }

        arabic_letters = [mapping.get(letter, letter) for letter in english_letters]
        return ' '.join(arabic_letters)

    def search_cars_by_plate(self, plate_characters):
        print(f"search_cars_by_plate: {plate_characters}")
        # Extract numbers from plate_characters
        plate_numbers = ''.join(filter(str.isdigit, plate_characters))
        english_letters = ''.join(filter(str.isalpha, plate_characters))
        print(f"search_cars_by_plate: {plate_numbers}")
        print(f"search_cars_by_plate: {english_letters}")

        
        # Query the 'Cars' collection
        cars_ref = self.db.collection('Cars').where('plateNumbers', '==', plate_numbers).where('englishLetters', '==', english_letters)
        
        # Get documents matching the query
        matching_cars = cars_ref.get()
        
        # Check if any matching document exists
        if not matching_cars:
            return None, None  # No matching documents found
        
        # Get the first matching document
        first_matching_car = matching_cars[0]
        
        # Get the data of the first matching car document
        car_data = first_matching_car.to_dict()
        
        # Return the entire matching car document
        return car_data.get('userId'), first_matching_car.id
    
    def add_plate(self, user_id, station_id, plate_characters, carId):
    # Separate plate numbers and letters
        plate_characters = plate_characters.upper()
        english_letters = ''.join(filter(str.isalpha, plate_characters))
        plate_numbers = ''.join(filter(str.isdigit, plate_characters))
        arabic_letters = self.english_to_arabic(english_letters)

        # Check if plate_numbers is empty
        if not plate_numbers:
            print("No numeric characters found in plate.")
            return
        
        # Check if the collection already exists, if not, create it
        plates_detected_ref = self.db.collection('Plates_Detected')
        if not plates_detected_ref.get():
            plates_detected_ref.document().create({})  # Create a dummy document
        
        # Add the information to the collection
        doc_ref = plates_detected_ref.document()
        # Add the information to the collection
        doc_ref = self.db.collection('Plates_Detected').document()
        doc_ref.set({
            'userId': user_id,
            'stationId': station_id,
            'plateNumbers': plate_numbers,
            'englishLetters': english_letters,
            'arabicLetters': arabic_letters,
            'carId': carId
        })





