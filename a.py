for i in range(1, 10):
    fn = "data/groundtruth/dissertation/q" + str(i) + '.txt'
    with open("./"+fn, "w") as f:
        f.write("hi")
