# fruit_manager.py

class FruitManager:
    def __init__(self):
        """Initialize the fruit stock as an empty dictionary."""
        self.fruit_stock = {}

    def add_fruit(self):
        """Add fruit to the stock."""
        fruit = input("Enter fruit name: ").strip().capitalize()
        if not fruit:
            print("Invalid input. Fruit name cannot be empty.")
            return

        try:
            quantity = int(input("Enter quantity: "))
            if quantity <= 0:
                print("Quantity must be greater than 0.")
                return

            if fruit in self.fruit_stock:
                self.fruit_stock[fruit] += quantity
            else:
                self.fruit_stock[fruit] = quantity
            
            print(f"✅ Successfully added {quantity} units of {fruit}.")
        except ValueError:
            print("Invalid quantity! Please enter a valid number.")

    def view_stock(self):
        """Display all available fruit stock."""
        print("\n📌 Current Fruit Stock:")
        if not self.fruit_stock:
            print("No stock available.")
        else:
            for fruit, quantity in self.fruit_stock.items():
                print(f"{fruit}: {quantity} units")

    def update_stock(self):
        """Update the stock quantity of an existing fruit."""
        fruit = input("Enter fruit name to update: ").strip().capitalize()
        if fruit in self.fruit_stock:
            try:
                quantity = int(input("Enter new quantity: "))
                if quantity < 0:
                    print("Quantity cannot be negative.")
                    return

                self.fruit_stock[fruit] = quantity
                print(f"✅ Successfully updated {fruit} stock to {quantity} units.")
            except ValueError:
                print("Invalid quantity! Please enter a valid number.")
        else:
            print(f"❌ {fruit} not found in stock.")
