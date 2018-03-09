import numpy as np
import util

target_shape = {'height':100, 'width':200}
error_margin = 5 # pixels

filter_size_list = [2, 3, 4, 5, 6]

minimum_net_depth = 3
maximum_net_depth = 9
stride = 2
starting_size = 1 # at deepest layer

def main():
    size_list = [[{'output_size': starting_size, 'filter_size': []}]]
    for depth in range(maximum_net_depth):
        children = []
        for node in size_list[-1]:
            for filter_size in filter_size_list:
                children.append({
                    'output_size': util.get_input_size(node['output_size'], filter_size, stride),
                    'filter_size': node['filter_size'] + [filter_size]})
        size_list.append(children)
    size_list = [size for size_same_depth in size_list for size in size_same_depth]

    for target_name, target_size in target_shape.items():
        print('Generating filter sizes for {}:'.format(target_name))
        size_array = np.array([size['output_size'] for size in size_list])
        for best in np.where(np.abs(size_array - target_size + error_margin / 2) < error_margin / 2)[0]:
            if len(size_list[best]['filter_size']) < minimum_net_depth:
                continue
            print('filter size sequence (deep to shallow): {}, input size: {}'.format(
                size_list[best]['filter_size'],
                size_list[best]['output_size']))

if __name__ == "__main__":
    main()