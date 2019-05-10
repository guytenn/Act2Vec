import ndjson
from utils.utils import printProgress
import numpy as np
from utils.corpus_utils import unbag_corpus


class PainterDataLoader:
    def __init__(self, path, max_size=None, stroke_length=5,
                 jump_thresh=(5, 10), save_jumps=False,
                 continuous=False, with_print=False):
        with open(path) as f:
            data = ndjson.load(f)
        self.stroke_length = stroke_length
        drawings = {'initial_pos': [], 'actions': []}
        # print("Found {} drawings.".format(len(data)))
        if max_size:
            N = min(max_size, len(data))
        else:
            N = len(data)
        for i in range(N):
            if with_print and i % int(N / 200) == 0:
                printProgress(i, N)
            actions = ['S'] * stroke_length
            last_end_pt = None
            prev_stroke_len = 0
            initial_pos = None
            for stroke in data[i]['drawing']:
                xy = list(zip(*stroke))
                initial_pos = (int(xy[0][0]), int(xy[0][1]))
                # when not at beginning of drawing
                if last_end_pt is not None:
                    # this is at least the second stroke, get the jump distance in x and y
                    jump_action = [xy[0][0] - last_end_pt[0], xy[0][1] - last_end_pt[1]]
                    # dont include drawings with jump distance larger than jump_thresh[1]
                    if jump_action[0] > jump_thresh[1] or jump_action[1] > jump_thresh[1]:
                        actions = []
                        break
                    # if the jump distance is smaller than jump_thresh[0], then just connect with a line
                    elif jump_action[0] <= jump_thresh[0] and jump_action[1] <= jump_thresh[0]:
                        line = PainterDataLoader.get_line(last_end_pt[0], xy[0][0], last_end_pt[1], xy[0][1])
                        actions += line['actions']
                        prev_stroke_len += len(line['actions'])
                    else:
                        # there was a jump between minimum threshold and maximum threshold
                        # in case we don't want to save jump actions, leave empty
                        if not save_jumps:
                            actions = []
                            break
                        # if we do save jump actions, then calculate the jump and add it as an action
                        # but before that, make sure that the previous stroke length is divisible by
                        # the word count
                        leftover_amount = prev_stroke_len % stroke_length
                        actions = self.add_leftovers(actions, jump_action, leftover_amount)
                        prev_stroke_len = 0
                # get actions from stroke
                for j in range(1, len(xy)):
                    line = PainterDataLoader.get_line(xy[j - 1][0], xy[j][0], xy[j - 1][1], xy[j][1])
                    actions += line['actions']
                    # counter of stroke length used to make sure all strokes lengths are divisible by stroke_length
                    prev_stroke_len += len(line['actions'])
                last_end_pt = xy[-1]
            if actions:
                # if you reached here without an empty list, then add 'E' (end) actions and
                # divide the action list into composite actions based on stroke_length
                actions += ['E'] * stroke_length
                drawings['initial_pos'].append(initial_pos)
                drawings['actions'].append(self.divide_into_composite(actions, continuous=continuous))

        self.drawings = drawings

    def get_images_as_embeddings(self, inds, buffer_size, act2vec_model):
        images = []
        vocab = list(act2vec_model.wv.vocab)
        blank_embedding = act2vec_model.wv['S' * len(vocab[0])]
        for i in range(len(inds)):
            actions = self.drawings['actions'][inds[i]]
            if len(actions) > buffer_size:
                continue
            image = np.ones((buffer_size, act2vec_model.wv.vector_size)) * blank_embedding
            tmp_img = act2vec_model.wv[actions]
            image[0:len(tmp_img), :] = tmp_img
            images.append(image)
        return np.array(images)


    def get_images(self, inds, width=256, height=256):
        images = np.zeros((len(inds), height, width, 1))

        for i in range(len(inds)):
            images[i], _ = PainterDataLoader.image_from_actions(self.drawings['actions'][inds[i]], width=width, height=height)

        return images

    def samples_images(self, N, width=256, height=256):
        inds = np.random.randint(len(self.drawings['actions']), size=N)
        return self.get_images(inds, width=width, height=height)

    def add_leftovers(self, actions, jump_action, leftover_amount):
        for action in actions[-leftover_amount:]:
            if action == 'U':
                jump_action[0] -= 1
            elif action == 'D':
                jump_action[0] += 1
            elif action == 'R':
                jump_action[1] += 1
            elif action == 'L':
                jump_action[1] -= 1
        return actions[0:-leftover_amount] + ["Jx{}y{}".format(*jump_action)]

    def divide_into_composite(self, actions: list, continuous=False):
        composite_actions = []
        ind = 0
        while ind < len(actions):
            curr_word = actions[ind: ind + self.stroke_length]
            if 'J' in curr_word and continuous:
                raise ValueError('Currently there is no support for jump actions in continuous embedding mode')
            if 'J' in curr_word[0]:
                composite_actions.append(curr_word[0])
                ind += 1
            else:
                composite_actions.append(''.join(actions[ind:ind+self.stroke_length]))
                if continuous:
                    ind += 1
                else:
                    ind += self.stroke_length
        if ('J' not in composite_actions[-1]) and len(composite_actions[-1]) < self.stroke_length:
            composite_actions = composite_actions[:-2]
        return composite_actions

    @staticmethod
    def image_from_actions(actions, width=256, height=256, center=False):
        image_uncropped = np.zeros((2*height, 2*width, 1))

        last_point = [height, width]
        image_uncropped[last_point[0], last_point[1], 0] = 1
        
        max_point = {'vert': last_point[0], 'horz': last_point[1]}
        min_point = {'vert': last_point[0], 'horz': last_point[1]}
        done = False

        for action_str in actions:
            if 'J' in action_str:
                raise ValueError("Jump actions are currently not supported")
            if done:
                break
            for a in action_str:
                if done:
                    break
                if a == 'S':
                    continue
                if a == 'E':
                    done = True
                    break
                if a == 'U':
                    last_point[0] -= 1
                    min_point['vert'] = min(min_point['vert'], last_point[0])
                elif a == 'D':
                    last_point[0] += 1
                    max_point['vert'] = max(max_point['vert'], last_point[0])
                elif a == 'R':
                    last_point[1] += 1
                    max_point['horz'] = max(max_point['horz'], last_point[1])
                elif a == 'L':
                    last_point[1] -= 1
                    min_point['horz'] = min(min_point['horz'], last_point[1])
                else:
                    raise ValueError("Unknown Action: " + a)
                if (max_point['horz'] - min_point['horz'] >= width) or \
                        (max_point['vert'] - min_point['vert'] >= height):
                    done = True
                    break


                image_uncropped[last_point[0], last_point[1], 0] = 1

        if center:
            w, h = int(width / 2), int(height /2)
            cropped_image = image_uncropped[min_point['vert'] - h: min_point['vert'] + h,
                            min_point['horz'] - w: min_point['horz'] + w, :]
        else:
            cropped_image = image_uncropped[min_point['vert']:min_point['vert'] + height,
                                            min_point['horz']:min_point['horz'] + width, :]
        return cropped_image, done

    @staticmethod
    def get_line(start_y, end_y, start_x, end_x):
        x1 = end_x
        y1 = end_y
        dx = abs(x1 - start_x)
        dy = abs(y1 - start_y)

        sx = int((start_x < x1)) - int((start_x >= x1))
        sy = int((start_y < y1)) - int((start_y >= y1))

        if sx == 1:
            a_horz = 'R'
        else:
            a_horz = 'L'

        if sy == 1:
            a_vert = 'D'
        else:
            a_vert = 'U'

        err = dx - dy

        line = dict()
        line["start"] = [start_x, start_y]
        line["actions"] = []
        line["end"] = [end_x, end_y]

        while True:
            if start_x == x1 and start_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                start_x = start_x + sx
                line["actions"].append(a_horz)
            elif e2 < dx:
                err = err + dx
                start_y = start_y + sy
                line["actions"].append(a_vert)

        return line


if __name__ == "__main__":
    import os

    path = 'QuickDraw/drawings_corpus/'
    datasets = ["square"]
    # datasets = ["banana", "basketball", "airplane", "ant", "angel", "barn", "basket", "baseball"]
    # datasets = "all"
    with_print = True
    bag_of_words = True
    continuous_embedding = False
    stroke_lengths = [10, 15, 20]
    embedding_dims = [[5, 10, 25],
                      [10, 15, 25, 50],
                      [15, 20, 30, 50, 100]] # embedding dims for each stroke length
    windows = [2, 3, 5]

    if datasets == "all":
        categories = os.listdir(path)
        datasets = [x.split('.')[0] for x in categories if x.endswith("ndjson")]
        with_print_dataset = False
    else:
        with_print_dataset = with_print

    for ww, stroke_length in enumerate(stroke_lengths):
        print("Stroke Length " + str(stroke_length))

        orig_corpus = []
        for dd, dataset in enumerate(datasets):
            if with_print:
                print("DATASET: " + dataset)
                printProgress(dd, len(datasets))

            full_path = os.path.join(path, dataset + ".ndjson")

            print('Creating corpus for dataset ' + dataset)
            DL = PainterDataLoader(full_path,
                                   stroke_length=stroke_length,
                                   continuous=continuous_embedding,
                                   with_print=with_print_dataset)

            orig_corpus += DL.drawings['actions']

        from gensim.test.utils import get_tmpfile
        from gensim.models import Word2Vec

        # corpus = DL.drawings['actions']
        path_w2vec = get_tmpfile("act2vec.model")

        for embedding_dim in embedding_dims[ww]:
            for window in windows:
                if not bag_of_words:
                    print('Unbagging corpus...')
                    corpus = unbag_corpus(orig_corpus, window=window)
                else:
                    corpus = orig_corpus

                print('Creating Act2Vec model...')
                model = Word2Vec(corpus, size=embedding_dim, window=window, min_count=1, workers=4)

                vocab = list(model.wv.vocab)

                print('total of {} actions in dictionary'.format(len(vocab)))

                print("Training act2vec model")
                model.train(corpus, total_examples=len(corpus), epochs=50)

                if len(datasets) > 1:
                    dataset_name = 'many'
                else:
                    dataset_name = '+'.join(datasets)

                desc = "act2vec_dataset:" + dataset_name + "_wordlen:" + str(stroke_length) + \
                        "_dim:" + str(embedding_dim) + "_win:" + str(window)

                if continuous_embedding:
                    desc += "_continuous"

                if bag_of_words:
                    desc += "_bagofwords"

                model.save("QuickDraw/embeddings/embeddings/" + desc + ".model")
