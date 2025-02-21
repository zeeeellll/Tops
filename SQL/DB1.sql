use DS;

CREATE TABLE Customer (
    customer_id INT PRIMARY KEY,
    cust_name VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    grade INT,
    salesman_id INT NOT NULL
);

INSERT INTO Customer (customer_id, cust_name, city, grade, salesman_id) VALUES
(3002, 'Nick Rimando', 'New York', 100, 5001),
(3007, 'Brad Davis', 'New York', 200, 5001),
(3005, 'Graham Zusi', 'California', 200, 5002),
(3008, 'Julian Green', 'London', 300, 5002),
(3004, 'Fabian Johnson', 'Paris', 300, 5006),
(3009, 'Geoff Cameron', 'Berlin', 100, 5003),
(3003, 'Jozy Altidor', 'Moscow', 200, 5007),
(3001, 'Brad Guzan', 'London', NULL, 5005);

CREATE TABLE Orders (
    ord_no INT PRIMARY KEY,
    purch_amt Int,
    ord_date DATE,
    customer_id INT,
    salesman_id INT 
);

INSERT INTO Orders (ord_no, purch_amt, ord_date, customer_id, salesman_id) VALUES
(70001, 150.50, '2012-10-05', 3005, 5002),
(70009, 270.65, '2012-09-10', 3001, 5005),
(70002, 65.26, '2012-10-05', 3002, 5001),
(70004, 110.50, '2012-08-17', 3009, 5003),
(70007, 948.50, '2012-09-10', 3005, 5002),
(70005, 2400.60, '2012-07-27', 3007, 5001),
(70008, 5760.00, '2012-09-10', 3002, 5001),
(70010, 1983.43, '2012-10-10', 3004, 5006),
(70003, 2480.40, '2012-10-10', 3009, 5003),
(70012, 250.45, '2012-06-27', 3008, 5002),
(70011, 75.29, '2012-08-17', 3003, 5007),
(70013, 3045.60, '2012-04-25', 3002, 5001);


CREATE TABLE Salesman (
    salesman_id INT,
    name VARCHAR(100),
    city VARCHAR(100) ,
    commission Int
);

INSERT INTO Salesman (salesman_id, name, city, commission) VALUES
(5001, 'James Hoog', 'New York', 0.15),
(5002, 'Nail Knite', 'Paris', 0.13),
(5005, 'Pit Alex', 'London', 0.11),
(5006, 'Mc Lyon', 'Paris', 0.14),
(5007, 'Paul Adam', 'Rome', 0.13),
(5003, 'Lauson Hen', 'San Jose', 0.12);


CREATE TABLE employee_details (
    EMP_IDNO INT PRIMARY KEY,
    EMP_FNAME VARCHAR(100),
    EMP_LNAME VARCHAR(100),
    EMP_DEPT varchar(100)
);

INSERT INTO employee_details (EMP_IDNO, EMP_FNAME, EMP_LNAME, EMP_DEPT) VALUES
(127323, 'Michale', 'Robbin', 57),
(526689, 'Carlos', 'Snares', 63),
(843795, 'Enric', 'Dosio', 57),
(328717, 'Jhon', 'Snares', 63),
(444527, 'Joseph', 'Dosni', 47),
(659831, 'Zanifer', 'Emily', 47),
(847674, 'Kuleswar', 'Sitaraman', 57),
(748681, 'Henrey', 'Gabriel', 47),
(555935, 'Alex', 'Manuel', 57),
(539569, 'George', 'Mardy', 27),
(733843, 'Mario', 'Saule', 63),
(631548, 'Alan', 'Snappy', 27),
(839139, 'Maria', 'Foster', 57);

select * from Customer;

select * from Orders;

select * from salesman;

select * from employee_details;



SELECT customer_id, cust_name, city, grade, salesman_id
FROM Customer;

select customer_id,cust_name,city,grade 
from Customer
where city = 'New York' and grade > 100;
 
select  customer_id, cust_name, city, grade, salesman_id
from customer
where city = 'New York' or grade > 100;

select customer_id,cust_name,city,grade,salesman_id
from Customer
where city = 'New York' or grade < 100;


-- From the following table, write a SQL query to identify customers who are not from the city of 'New York' and do 
-- not have a grade value greater than 100. Return customer_id, cust_name, city, grade, and salesman_id.
-- Sample table: customer

use ds;

select customer_id,cust_name,city,grade,salesman_id
from customer
where city <>'New york' and grade < 100;

-- From the following table, write a SQL query to find details of all orders excluding those with ord_date equal to 
-- '2012-09-10' and salesman_id higher than 5005 or purch_amt greater than 1000.Return ord_no, purch_amt, ord_date, customer_id
-- and salesman_id.


select ord_no, purch_amt, ord_date, customer_id, salesman_id
from Orders
where ord_date = '2012-09-10' and salesman_id > 5055 or purch_amt > 1000;

/*From the following table, write a SQL query to find the details of those
 salespeople whose commissions range from 0.10 to0.12. Return salesman_id, 
name, city, and commission.*/

show tables;
select * from salesman;

select salesman_id,name,commission
from salesman 
where commission between 0.10 and 0.12;

/*From the following table, write a SQL query to find details of all orders 
with a purchase amount less than 200 or exclude orders with an order date greater than or equal to 
'2012-02-10' and a customer ID less than 3009. Return ord_no, purch_amt, ord_date, customer_id and salesman_id.*/

show tables;
select * from customer;
select * from orders;

select ord_no, purch_amt, ord_date, customer_id, salesman_id
from orders
where purch_amt < 200 or ord_date  >= '2012-02-10' and customer_id < 3009;

/*From the following table, write a SQL query to find all orders that meet the following conditions. 
Exclude combinations of order date equal to '2012-08-17' or customer ID greater than 3005 and purchase amount less than 1000.*/

select * from orders
where ord_date = '2012-08-17' or customer_id > 3005 and purch_amt < 1000;

/*Write a SQL query that displays order number, purchase amount, and the achieved and unachieved percentage (%) 
for those orders that exceed 50% of the target value of 6000.*/

SELECT 
    ord_no, 
    purch_amt, 
    (purch_amt / 6000) * 100 AS achieved_percentage,
    (100 - (purch_amt / 6000) * 100) AS unachieved_percentage
FROM orders
WHERE purch_amt > 3000;

-- From the following table, write a SQL query to find the details of all employees whose last name is ‘Dosni’ or ‘Mardy’.
-- Return emp_idno, emp_fname, emp_lname, and emp_dept.

show tables;
select * from employee_details;
select emp_fname,emp_lname,emp_dept
from employee_details
where emp_lname = 'Dosni' or 'Mardy';

-- the following table, write a SQL query to find the employees who work at
-- depart 47 or 63. Return emp_idno, emp_fname, emp_lname, and emp_dept.

select emp_idno, emp_fname, emp_lname,emp_dept 
from employee_details
where emp_dept in(47,63);

use ds;


-- From the following table, write a SQL query to calculate total purchase amount of all orders. Return total purchase amount.

SELECT SUM(purch_amt) AS total_purchase_amount
FROM orders;

-- From the following table, write a SQL query to calculate the average purchase amount of all orders. Return average purchase amount.

select avg(purch_amt) from orders;

-- From the following table, write a SQL query that counts the number of unique salespeople. Return number of salespeople

select * from salesman;

use Ds;
show tables;

select * from salesman;
-- create a view for those salespeople who belong to the city of New York.

create view n_S as 
SELECT name 
from salesman
where city = 'New York'; 

select * from n_s;

-- create a view for all salespersons. Return salesperson ID, name, and city.

create view s_p as
select salesman_id,name,city from salesman;

select * from s_p;

-- create a view to locate the salespeople in the city 'New York'.

create view n_S1 as 
SELECT name 
from salesman
where city in('New York'); 

select * from n_s1;

-- create a view that counts the number of customers in each grade.
select * from customer;

CREATE VIEW CustomerGradeCount AS
SELECT grade, COUNT(customer_id) AS customer_count
FROM customer
GROUP BY grade;

select * from CustomerGradeCount;


-- create a view to count the number of unique customers, compute the average and the total purchase amount of customer orders by each date.
CREATE VIEW DailySalesSummary AS
SELECT 
    ord_date, 
    COUNT(DISTINCT customer_id) AS unique_customers, 
    AVG(purch_amt) AS average_purchase, 
    SUM(purch_amt) AS total_purchase
FROM orders
GROUP BY ord_date;

select * from DailySalesSummary;
select * from orders;

-- create a view to get the salesperson and customer by name. Return order name, purchase amount, salesperson ID, name, customer name.
select * from customer;

create view new_seq as 
select o.ord_no,o.purch_amt,o.salesman_id,c.cust_name
from orders o
join customer c on o.salesman_id = c.salesman_id;


select * from new_seq;

drop view new_seq;
use ds;
CREATE VIEW SalesDetails AS
SELECT 
    o.ord_no,
    o.purch_amt,
    s.salesman_id,
    s.name,
    c.cust_name
FROM 
    orders o
JOIN 
    salesman s ON o.salesman_id = s.salesman_id
JOIN 
    customer c ON o.customer_id = c.customer_id;

select * from SalesDetails;


use ds;

-- create a view to find the salesperson who handles a customer who makes the highest order of the day. Return order date, salesperson ID, name.

select * from orders;

select o.ord_date,s.salesman_id,s.name
from salesman s
join orders o on s.salesman_id = o.salesman_id
WHERE o.purch_amt = (
    SELECT MAX(purch_amt)
    FROM Orders o2
    WHERE o2.ord_date = o.ord_date
)
order by salesman_id asc;

use ds;

-- create a view to count the number of salespeople in each city. Return city, number of salespersons.

select city,count(salesman_id)
from salesman 
group by city;

-- create a view to find all the customers who have the highest grade. Return all the fields of customer.
select * from customer;

create view a as 
select *
from customer
where grade = (
	select max(grade) from customer);

-- create a view to find the salesperson who deals with the customer with the highest order at least three times per day.
-- Return salesperson ID and name.

create view i as
select distinct salesman_id, name
from salesman a
where 3<=(select count(*)
from we b
where a.salesman_id = b.salesman_id);
select * from i;

select * from TopSalespeopleFrequent;
use hrdb;


with dep_sal as(
select departmenr_id,sum(salary) as total_sal
from employees
group by department_id)
select d.department_id_name,ds.total_salary
from departments d
join dap_sal on d.department_id = ds.depaerment_id;







