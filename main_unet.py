import numpy as np
import util

target_shape = {'height':512, 'width':512}
error_margin = 0 # pixels

filter_size_list = [3, 4, 5]

minimum_net_depth = 8
maximum_net_depth = 10
stride = 2
starting_size_list = range(1, 5) # at deepest layer

def main():
    sizes = {}
    for starting_size in starting_size_list:
        sizes[starting_size] = {}
        children = [{'output_size': starting_size, 'filter_size': []}]
        for depth in range(maximum_net_depth, -1, -1):
            sizes[starting_size][depth] = children
            children = []
            for node in sizes[starting_size][depth]:
                for filter_size in filter_size_list:
                    children.append({
                        'output_size': util.get_input_size(node['output_size'], filter_size, stride),
                        'filter_size': node['filter_size'] + [filter_size]})

    for target_name, target_size in target_shape.items():
        print('Generating filter sizes for {}:'.format(target_name))
        for depth in range(maximum_net_depth - minimum_net_depth + 1):
            size_array = np.array([node['output_size'] for node in sizes[1][depth]])
            for best in np.where(np.abs(size_array - target_size + error_margin / 2) <= error_margin / 2)[0]:
                if len(sizes[1][depth][best]['filter_size']) < minimum_net_depth:
                    continue
                print('filter size sequence (deep to shallow): {}, proper input sizes: {}'.format(
                    sizes[1][depth][best]['filter_size'],
                    [sizes[s][depth][best]['output_size'] for s in starting_size_list]))

if __name__ == "__main__":
    main()