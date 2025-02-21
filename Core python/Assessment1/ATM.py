class assisment1:
    pin = 0
    def __init__(self):
        #self.balance()
        self.set_pin()



    def menu(self):
        print('''
            Hello welcome to the ATM service, how can i help you ?

              1. Press 1 for change pin,
              2. Press 2 for check balance,
              3. Press 3 for deposite amount, 
              4. Press 4 for widthdraw amount,
              5. press 5 for exit 
            ''')
        
        user_input = input("Enter Youe choice : ")

        if user_input == '1':
            self.change_pin()
        elif user_input == '2':
            pass
        elif user_input == '3':
            pass
        elif user_input == '4':
            pass
        else :
            exit()

    def set_pin(self):
        global pin
        create_pin = input("Enter Your Pin to create : ")
  
        if len(create_pin) == 4:
            pin = create_pin
            print("Your pin has been create successfully")
            print(f"Your pin is {pin}")
            self.menu()
        else:
            print("Enter only 4 digits")


    def change_pin():
        global pin
        pinn = int(input("Enter Your pin : "))
        
        if pinn == pin:
            new_pin = input("Enter new pin : ")
            if len(new_pin) == 4:
                pin = new_pin
                print("Your pin has been create successfully")
                print(f"Your pin is {pin}")
                self.menu()
            else:
                print("Enter only 4 digits")
        else:
            print("Inncorrect pin")
            self.menu()
            
    
obj = assisment1()

