
def __main__():

	outfile = open('symmetry.csv', 'w')
	for i in range(1, 2):
		infile = open(f'symmetry_{i}', 'r')
		for line in infile.readlines()[1:]:
			outfile.write(line)
		infile.close()

	outfile.close()
	file1 = open('OLD.csv', 'r')
	file2 = open('symmetry.csv', 'r')

	n = []
	for line in range(136):
		n.append(file1.readline() == file2.readline())

	print(sum(n))
	file1.close()
	file2.close()
if __name__ == "__main__":
	__main__()
