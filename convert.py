import csv

# ANSI escape codes for colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

# Function to format a row as a table row with colors
def format_row(row, widths):
    formatted_row = "|"
    for i, cell in enumerate(row):
        # Add color to cells based on their content
        if cell.strip().isdigit():
            formatted_row += f" {COLOR_GREEN}{cell.strip():<{widths[i]}}{COLOR_RESET} |"
        else:
            formatted_row += f" {COLOR_RED}{cell.strip():<{widths[i]}}{COLOR_RESET} |"
    return formatted_row + "\n"

# Read the CSV file and format its content
with open('covid_global.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# Calculate maximum widths for each column
widths = [max(len(cell.strip()) for cell in column) for column in zip(*rows)]

# Write the formatted content to a text file
with open('output.txt', 'w', encoding='utf-8') as txtfile:
    txtfile.write("|----|-------------------|---------|------------|---------|----------|----------|\n")
    txtfile.write(format_row(rows[0], widths))
    txtfile.write("|----|-------------------|---------|------------|---------|----------|----------|\n")
    for row in rows[1:]:
        txtfile.write(format_row(row, widths))
    txtfile.write("|----|-------------------|---------|------------|---------|----------|----------|\n")
