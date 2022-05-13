import os


# @dict:        Dictionary containing centrality scores for network nodes
def produce_report(dict):
    report = ''
    for key, value in dict.items():
        report += f"{key}:\t{value}\n"

    with open('output.txt', 'w', encoding="utf-8") as file:
        file.write(report)

    top_five_list = list()
    try:
        top_five_list = sorted(dict, key=dict.get, reverse=True)[:5]
    except:
        print('oops, something went wrong!')

    print("Top five nodes: ", end='')
    total = 0
    for i, item in enumerate(top_five_list):
        if i == len(top_five_list) - 1:
            total += dict[item]
            print(f'{item}', end='')
        else:
            total += dict[item]
            print(f'{item}, ', end='')

    print(f'\nAverage of top five is: {total / len(top_five_list)}')
