from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# constraint satisfaction problem
class CSP:
    def __init__(
        self,
        variables: list[str],
        domains: dict[str, set[int]],
        constraints: dict[str, list[str]],
    ):
        """_summary_

        Args:
            variables (list[str]): list of region names
            domains (dict[str, set[int]]): information about what colors are available for each region
            constraints (dict[str, list[str]]): dict showing information about neighbours
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution: Union[dict[str, int], None] = None

    def solve(self) -> Union[dict[str, int], None]:
        """_summary_

        Returns:
            Union[dict[str, int], None]: solution to the problem or None if not found
        """
        # create initial solution state (empty)
        assignment: dict[str, int] = {}
        self.solution = self.backtrack(assignment)
        return self.solution

    def select_unassigned_variable(
        self, assignment: dict[str, int]
    ) -> Union[str, None]:
        """Chooses the next variable (region) to assign a color to, based on the current state of the assignment

        Args:
            assignment (dict[str, int]): current state of the solution

        Returns:
            Union[str, None]: first found unassigned variable
        """
        # for each region on the map
        for var in self.variables:
            # return it if it was not already assigned a value
            if var not in assignment:
                return var

        # there are no uncolored regions left - return None
        return None

    def is_consistent(self, var: str, value: int, assignment: dict[str, int]) -> bool:
        """Checks if assigning a given value (color) to a variable (region) is
        consistent with the current assignment, ensuring no neighbors share the same color

        Args:
            var (str): name of the region to check
            value (int): color of the checked region
            assignment (dict[str, int]): current color assignment on the map

        Returns:
            bool: whether the given color assignment fulfills the problem constraints
        """
        # for each neighbour of the selected region
        for neighbor in self.constraints[var]:
            # if they share the same color - return false
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        # no collision found - return true
        return True

    def forward_checking(
        self, var: str, value: int, assignment: dict[str, int]
    ) -> dict[str, set[int]]:
        """After assigning a color to a region, this method removes that color
        from the domains of all neighboring regions to prevent them from bein
        assigned the same color. It keeps track of the removed values to
        restore them if necessary.

        Args:
            var (str): name of region which recently got its color assignment
            value (int): color assigned to the region
            assignment (dict[str, int]): current color assignment on the map

        Returns:
            dict[str, set[int]]: dictionary containing information about colors removed from the domain
        """
        # create a dictionary with keys which are only the regions not yet colored
        removed_values: dict[str, set[int]] = {
            v: set() for v in self.variables if v not in assignment
        }
        # for each neighbour of the selected region
        for neighbor in self.constraints[var]:
            # if the neighbour is not colored yet
            if neighbor not in assignment:
                # if the color of the current region is in the domain of the neighbour
                if value in self.domains[neighbor]:
                    # remove the color from neighbours domain
                    self.domains[neighbor].remove(value)
                    # save the information about the removal
                    removed_values[neighbor].add(value)

        return removed_values

    def backtrack(self, assignment: dict[str, int]) -> Union[dict[str, int], None]:
        """perform backtracking step to find the solution recursively

        Args:
            assignment (dict[str, int]): current solution state

        Returns:
            Union[dict[str, int], None]: next solution state or None
        """
        # recursive base case
        # if the solution (assignment) contains all variables then we found the solution
        if len(assignment) == len(self.variables):
            return assignment
        
        # find next region to assign color to
        var = self.select_unassigned_variable(assignment)
        
        # check for edge cas
        if var is None:
            return None
        
        # for each possible color for the current region
        for value in self.domains.get(var, set()):
            # check if this color assignment is possible
            if self.is_consistent(var, value, assignment):
                # color the region
                assignment[var] = value
                # perform forward checking and save the removed values
                removed_values = self.forward_checking(var, value, assignment)
                # perform next recursive step
                result = self.backtrack(assignment)
                
                # the solution was found in the recursive call
                if result is not None:
                    return result
                
                # the solution was not found in the recursive call
                del assignment[var] # revert the color assignment
                # restore domains to the state before forward checking
                for v, vals in removed_values.items():
                    self.domains[v] |= vals
                    
        # none of the color chosen fulfills the constraints - return None
        return None


def validate_map(map: dict[str, list[str]]) -> bool:
    """Check if given map contains proper data
    The relationships should be declared both ways
    The region cannot be adjacent to itself

    Args:
        map (dict[str, list[str]]): dict containing adjacency information

    Returns:
        bool: result whether the map is valid or not
    """
    # Iterate through each key and its adjacent nodes
    for region_name, neighbors in map.items():
        # Check if a region is listed as adjacent to itself
        if region_name in neighbors:
            return False
        for neighbor in neighbors:
            # Check if the current key is in its neighbor's adjacency list
            if region_name not in map.get(neighbor, []):
                return False
    return True


def csp_factory(map: dict[str, list[str]], num_colors: int) -> Union[CSP, None]:
    """Create the CSP object initializing it with proper values

    Args:
        map (dict[str, list[str]]): dict showing neighbour relations
        num_colors (int): num colors to use during the coloring

    Returns:
        CSP: initialized CSP object
    """
    if validate_map(map) == False:
        return None
    
    # names of regions
    variables = list(map.keys())
    # sets of possible solutions for each regions
    domains = {region: set(range(num_colors)) for region in variables}
    # relations between neighbours
    constraints = map

    return CSP(variables, domains, constraints)


def circular_layout(variables):
    angle = 2 * np.pi / len(variables)
    return {
        v: (0.5 + 0.4 * np.cos(i * angle), 0.5 + 0.4 * np.sin(i * angle))
        for i, v in enumerate(variables)
    }


def plot_solution(map, solution):
    positions = circular_layout(solution.keys())
    fig, ax = plt.subplots(figsize=(8, 8))
    for region, pos in positions.items():
        color_index = solution[region] if region in solution else 0
        ax.add_patch(patches.Circle(pos, 0.05, color=plt.cm.tab20(color_index)))
        ax.text(pos[0], pos[1], region, ha="center", va="center", fontsize=9)
    for region, neighbors in map.items():
        for neighbor in neighbors:
            if neighbor in positions:
                start, end = positions[region], positions[neighbor]
                ax.plot([start[0], end[0]], [start[1], end[1]], "k-", lw=0.5)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def main():
    # this dictionary shows us which nodes of the graph will be neighbours
    # each node lists its neighbours
    cmap = {
        "ab": ["bc", "nt", "sk"],
        "bc": ["yt", "nt", "ab"],
        "mb": ["sk", "nu", "on"],
        "nb": ["qc", "ns", "pe"],
        "ns": ["nb", "pe"],
        "nl": ["qc"],
        "nt": ["bc", "yt", "ab", "sk", "nu"],
        "nu": ["nt", "mb"],
        "on": ["mb", "qc"],
        "pe": ["nb", "ns"],
        "qc": ["on", "nb", "nl"],
        "sk": ["ab", "mb", "nt"],
        "yt": ["bc", "nt"],
    }

    # User input for number of colors
    num_colors = 3

    csp = csp_factory(cmap, num_colors)
    if csp is None:
        print("Provided map is not valid")
        return
    
    sol = csp.solve()

    if sol:
        print(sol)
        plot_solution(cmap, sol)
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
