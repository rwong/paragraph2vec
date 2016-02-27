#!/usr/bin/env python3

import os
import unittest
import tempfile

from collections import Counter
from load_data import load_imdb_raw

class TestLoadData(unittest.TestCase):
    def _writeImdbReview(self, tempd, pars, dirs, rating_interval):
        rating_range = rating_interval[1] - rating_interval[0]
        for ndx, par in enumerate(pars):

            rating = (ndx % rating_range) + rating_interval[0] \
                     if rating_range > 0 else 0

            for directory in dirs:
                fname = tempd + '/' + directory + '/' + \
                        str(ndx + 1) + '_' + str(rating) + '.txt'

                with open(fname, 'w') as doc:
                    doc.write(par)

    def testLoadImdbRaw(self):
        # Document contents
        neg_pars = [ ('aaa1 '*20 + '. <br><bt> ' +
                      'aaa2 '*20 + '. <br/><br/> ')*3 ]*3
        pos_pars = [ ('bbb1 '*20 + '<br /><br />' +
                      'bbb2 '*20 + '.<br/><br/>')*3 ]*4
        unsup_pars = [ ('ccc1 '*20 + '.<br /><br /> ' +
                        'ccc2 '*20 + '. <br/><br/> ')*3 ]*3

        with tempfile.TemporaryDirectory() as tempd:
            neg_dirs = [ 'test/neg', 'train/neg', ]
            pos_dirs = [ 'test/pos', 'train/pos', ]
            unsup_dirs = [ 'train/unsup' ]
            for dirname in neg_dirs + pos_dirs + unsup_dirs:
                os.makedirs(tempd + '/' + dirname, 0o700)

            #print(tempd)

            neg_inter, pos_inter, unsup_inter = (1, 4), (7, 10), (0, 0)
            self._writeImdbReview(tempd, neg_pars, neg_dirs, neg_inter)
            self._writeImdbReview(tempd, pos_pars, pos_dirs, pos_inter)
            self._writeImdbReview(tempd, unsup_pars, unsup_dirs, unsup_inter)

            testset, trainset, unsupset = load_imdb_raw(tempd)

            # Based on mock file creation, should have a fixed number of
            # files with the same id, one for negative, one for positive;
            # otherwise wind up with duplicate data entries
            file_ids = Counter()
            for [ file_id, rating, file_contents, label ] in testset:
                self.assertTrue( file_ids[ file_id ] <= 1 )
                file_ids[ file_id ] += 1

                is_neg = neg_inter[0] <= rating <= neg_inter[1]
                is_pos = pos_inter[0] <= rating <= pos_inter[1]
                self.assertTrue( is_neg or is_pos )
                self.assertEqual( 0 if is_neg else 1, label )

                expected_contents = neg_pars[0] if is_neg else pos_pars[0]
                self.assertEqual( expected_contents, file_contents )

            file_ids = Counter()
            for [ file_id, rating, file_contents, label ] in trainset:
                self.assertTrue( file_ids[ file_id ] <= 1 )
                file_ids[ file_id ] += 1

                is_neg = neg_inter[0] <= rating <= neg_inter[1]
                is_pos = pos_inter[0] <= rating <= pos_inter[1]
                self.assertTrue( is_neg or is_pos )
                self.assertEqual( 0 if is_neg else 1, label )

                expected_contents = neg_pars[0] if is_neg else pos_pars[0]
                self.assertEqual( expected_contents, file_contents )

            # Things are a little different for unsupervised set
            file_ids = Counter()
            for [ file_id, rating, file_contents, label ] in unsupset:
                self.assertTrue( file_ids[ file_id ] == 0 )
                file_ids[ file_id ] += 1

                self.assertEqual( 0, 0 )
                self.assertEqual( -1, label )
                self.assertEqual( unsup_pars[0], file_contents )

if __name__ == '__main__':
    unittest.main()
