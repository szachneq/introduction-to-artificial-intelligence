# Constraint satisfaction problem
class CSP: 
	def __init__(self, variables, domains, constraints): 
		"""
		Initialization of the CSP class

		Parameters:
		- variables
		- domains
		- constraints
		"""
		self.variables = variables 
		self.domains = domains 
		self.constraints = constraints 
		self.solution = None
        
	def solve(self): 
		assignment = {} 
		self.solution = self.backtrack(assignment) 
		return self.solution

	#  Implement additional functions here
        
	def forward_checking(self, var, value, assignment):
		"""
		Function that removes the value from the domains of free variables that are in the constraints of the var

		Parameters:
		- var: variable that was assigned the value
		- value: value that was assigned to the variable
		- assignment: dict with all the assignments to the variables

		Returns:
		- removed_values: set of free variables from domains of which the value was removed

		"""
		# Your code starts here


	def backtrack(self, assignment): 
		"""
		Backtracking algorithm

		Parameters:
		- assignment: dict with all the assignments to the variables

		Returns:
		- assignment: dict with all the assigments to the variables, or None if solution is not found. Return the first found solution
		"""
		# Your code starts here

# Example of the input
cmap = {}
cmap["ab"] = ["bc","nt","sk"]
cmap["bc"] = ["yt", "nt", "ab"]
cmap["mb"] = ["sk","nu","on"]
cmap["nb"] = ["qc", "ns", "pe"]
cmap["ns"] = ["nb", "pe"]
cmap["nl"] = ["qc"]
cmap["nt"] = ["bc", "yt", "ab", "sk", "nu"]
cmap["nu"] = ["nt", "mb"]
cmap["on"] = ["mb", "qc"]
cmap["pe"] = ["nb", "ns"]
cmap["qc"] = ["on", "nb", "nl"]
cmap["sk"] = ["ab", "mb", "nt"]
cmap["yt"] = ["bc", "nt"]

# Take as the input from the user number of colors to use

# Based on the input and number of colors, create variables, domains, and constraints for initialization of CSP class


csp = CSP(variables, domains, constraints) 
sol = csp.solve()

print(sol)
