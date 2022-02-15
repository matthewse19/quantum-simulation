import qUtilities as qu

def main():
    #number of trials
    n = 10000
    results = []
    for _ in range(n):    
        ketPsi = qu.uniform(2)
        
        left_hand = ketPsi[0] * ketPsi[3]
        right_hand = ketPsi[1] * ketPsi[2]
        results.append(abs(left_hand - right_hand))

    with open('results.txt', 'w') as f:
        for result in results:
            f.write(str(result) + '\n')

if __name__ == "__main__":
    main()