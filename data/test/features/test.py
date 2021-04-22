
def __main__():

	outfile = open('symmetry.csv', 'w')
	outfile.write("image_id,symmetry\n")
	for i in range(1, 2):
		infile = open(f'symmetry_{i}.csv', 'r')
		for line in infile.readlines()[1:]:
			outfile.write(line)
		infile.close()

	outfile.close()
	file1 = open('OLD.csv', 'r')
	file2 = open('symmetry.csv', 'r')

	n = []
	for line in range(136):
		 
		n.append(file1.readline().split(',')[0] == file2.readline().split(',')[0])
		#print(file1.readline().split(','),file2.readline().split(',')[0])
	print(sum(n))
	file1.close()
	file2.close()
if __name__ == "__main__":
	__main__()
