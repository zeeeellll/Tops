class assisment1:
    
    def __init__(self):
        self.pin = None
        self.balance = 0

        self.creat_pin()  

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
            self.check_balance()
        elif user_input == '3':
            self.deposite_amt() 
        elif user_input == '4':
            self.withdraw_amt()
        elif user_input == '5':
            exit()
        else:
            print("Enter valid choice !")
            self.menu()

    def creat_pin(self):
        pinn = input("Enter pin to create : ")
        if(len(pinn)) == 4:
            self.pin = pinn
            print("Your pin is : ",self.pin)
            self.menu()
        else:
            print("Enter 4 digit number")


    def change_pin(self):
        count = 3
        for _ in range(3):
            pinn = input("Enter your pin : ")
            if(pinn == self.pin):
                new_pin = input("Enter new pin : ")
                if(len(new_pin)) == 4:
                    self.pin = new_pin
                    print("Your new pin is",self.pin)
                    self.menu()
                else:
                    print("Enter 4 digit number")
                    break
            else:
                count-=1
                if count > 1:
                    print(f"Incorrect pin, you have {count} attempts")
                elif count == 1 :
                    print("Incorrect pin, you have one last attempt")
                else:
                    print("Try again after some time")
                    self.menu()

    def check_balance(self):
        count = 3
        for _ in range(3):
            pinn = input("Enter your pin : ")
            if(pinn == self.pin):
                if(self.balance == 0):
                    print(f"Your balance is :{self.balance}")
                    self.menu()
                else:
                    print(f"Your balance is : {self.balance}")
                    self.menu()
                    break
            else:
                count-=1
                if count > 1:
                    print(f"Incorrect pin, you have {count} attempts")
                elif count == 1 :
                    print("Incorrect pin, you have one last attempt")
                else:
                    print("Try again after some time")
                    self.menu()
            
    def deposite_amt(self):
        count = 3
        for _ in range(3):
            pinn = input("Enter your pin : ")
            if(pinn == self.pin):
                amt = int(input("Enter amount to deposite : "))
                if(amt > 0):
                    self.balance+=amt
                    print(f"{amt} has been deposite successfully ")
                    print(f"Yout total amount is : {amt}")
                    self.menu()
                else:
                    print("Enter valid amount")
                    self.deposite_amt()
                break
            else:
                count-=1
                if count > 1:
                    print(f"Incorrect pin, you have {count} attempts")
                elif count == 1 :
                    print("Incorrect pin, you have one last attempt")
                else:
                    print("Try again after some time")
                    self.menu()

    def withdraw_amt(self):
        count = 3
        for _ in range(3):
            pinn = input("Enter your pin : ")
            if(pinn == self.pin):
                amt = int(input("Enter amount to Withdraw : "))
                if(amt > 0):
                    if(self.balance <= 0):
                        print("Insufficient balance in your account !")
                        self.menu()
                        break
                    else:
                        self.balance-=amt
                        print(f"{amt} has been Withdraw successfully ")
                        print(f"Yout total amount is : {amt}")
                        self.menu()
                        break
                else:
                    print("Enter valid amount !")
                    self.menu()
            else:
                count-=1
                if count > 1:
                    print(f"Incorrect pin, you have {count} attempts")
                elif count == 1 :
                    print("Incorrect pin, you have one last attempt")
                else:
                    print("Try again after some time")
                    self.menu()

obj = assisment1()