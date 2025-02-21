create database o;
use o;

CREATE TABLE Employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50),
    department VARCHAR(50),
    salary DECIMAL(10,2)
);

INSERT INTO Employees (emp_id, name, department, salary) VALUES
(1, 'Alice', 'HR', 50000),
(2, 'Bob', 'HR', 60000),
(3, 'Charlie', 'IT', 70000),
(4, 'David', 'IT', 75000),
(5, 'Eve', 'IT', 80000),
(6, 'Frank', 'Finance', 55000),
(7, 'Grace', 'Finance', 65000);

select * from employees;

SELECT 
    emp_id, 
    name, 
    department, 
    salary, 
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank_in_department
FROM Employees;


SELECT 
    emp_id, 
    name, 
    department, 
    salary, 
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rank_in_department
FROM Employees;


