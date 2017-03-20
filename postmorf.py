import re


MORPH_DELIMITER = '=='


def pure_morph_tok(dirty_token):
    '''
    :param dirty_token:
        palume    palu+me //_V_ me, //
    :return:
        palu==me
        None iff there's no morph token (####)
    '''

    mtok = dirty_token.split(' ' * 4)[1]
    if '####' in mtok:
        return None
    else:
        # throw away morph info, keep morphed word
        morphed_word = re.match('(.+?) //.*', mtok).group(1)
        # get rid of +0's
        morphed_word = morphed_word.rstrip('+0')
        # substitute own delimiter in place of morpher's
        clean_morph_word = re.sub('[+_=]', MORPH_DELIMITER, morphed_word)
        #print('{} --> {}'.format(mtok, clean_morph_word))
        return clean_morph_word


if __name__ == '__main__':

    with open('train.et', encoding='utf-8') as f:
        sents = f.readlines()
    with open('etana-out.et', encoding='utf-8') as f:
        morph_toks = f.readlines()

    new_sents = []
    morph_tok_idx = 0

    for sent in sents:
        words = sent.strip().split()
        new_words = []
        i = 0
        while i < len(words):
            #print(word)
            morphed = pure_morph_tok(morph_toks[morph_tok_idx])
            morph_tok_idx += 1
            if morphed is None:
                # no morph
                new_words.append(words[i])
            else:
                new_words.append(morphed)
                i += morphed.count(' ')  # morphezzor sometimes decides to group words; this keeps them from misaligning
            i += 1

        new_sents.append(' '.join(new_words))

    with open('etana-out.et', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(new_sents))
