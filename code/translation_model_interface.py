'''
Each class of translator model should be inherited from this interface.
'''
class ITranslationModel(object):

    def make_initial_state(self, lines):
        '''
        Accepts array of lines.
        Returns initial translation state for lines.
        '''
        raise Exception('Not implemented')

    def get_next_state_and_logits(self, state, outputs):
        '''
        Accepts current translation state and model outputs.
        Returns next translation state and logits.
        '''
        raise Exception('Not implemented')

    def get_output_tokenizer(self):
        '''
        Return output tokenizer used in model.
        '''
        raise Exception('Not implemented')
