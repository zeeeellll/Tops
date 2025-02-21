# main.py

from fruit_manager import FruitManager

def main():
    """Main function to run the Fruit Store Console Application."""
    fruit_manager = FruitManager()

    while True:
        print("\n📌 WELCOME TO FRUIT MARKET 📌")
        print("1) Manager")
        print("2) Customer (Feature Coming Soon)")
        print("3) Exit")

        role = input("Select your Role (1/2/3): ").strip()

        if role == "1":
            while True:
                print("\n📌 Fruit Market Manager Menu 📌")
                print("1) Add Fruit Stock")
                print("2) View Fruit Stock")
                print("3) Update Fruit Stock")
                print("4) Back to Main Menu")

                choice = input("Enter your choice: ").strip()
                

                if choice == "1":
                    fruit_manager.add_fruit()
                elif choice == "2":
                    fruit_manager.view_stock()
                elif choice == "3":
                    fruit_manager.update_stock()
                elif choice == "4":
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

        elif role == "2":
            print("\n👨‍👩‍👦 Customer functionality is under development.")

        elif role == "3":
            print("👋 Exiting program. Thank you!")
            break

        else:
            print("❌ Invalid input. Please select a valid option.")

if __name__ == "__main__":
    main()

