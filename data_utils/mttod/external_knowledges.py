"""
   MTTOD: external_knowledges.py

   Implements MultiWoZ JSON Object Handler.

   This code is referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/db_ops.py)

   Copyright 2021 ETRI LIRS, Yohan Lee
   Copyright 2019 Yichi Zhang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import random
from collections import defaultdict

from data_utils.mttod.utils import definitions
from data_utils.mttod.utils.io_utils import load_json
from utils.utils import get_or_create_logger


logger = get_or_create_logger(__name__)


class MultiWozDB:
    """ MultiWoZ JSON Handler class """
    def __init__(self, db_dir):
        self.dbs = {}

        for domain in definitions.ALL_DOMAINS:
            self.dbs[domain] = load_json(os.path.join(db_dir,
                                                      "{}_db_processed.json".format(domain)))

        self.db_domains = ["attraction", "hotel", "restaurant", "train"]

        extractive_ontology = {}
        for db_domain in self.db_domains:
            extractive_ontology[db_domain] = defaultdict(list)

            dbs = self.dbs[db_domain]
            for ent in dbs:
                for slot in definitions.EXTRACTIVE_SLOT:

                    if slot in ["leave", "arrive"]:
                        continue

                    if slot not in ent or ent[slot] in extractive_ontology[db_domain][slot]:
                        continue

                    extractive_ontology[db_domain][slot].append(ent[slot])

        self.extractive_ontology = extractive_ontology

    def one_hot_vector(self, domain, num):
        """Return number of available entities for particular domain."""
        vector = [0, 0, 0, 0]
        if num == '':
            return vector
        if domain != 'train':
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num == 1:
                vector = [0, 1, 0, 0]
            elif num <= 3:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        else:
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num <= 5:
                vector = [0, 1, 0, 0]
            elif num <= 10:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        return vector

    def addBookingPointer(self, turn_da):
        """Add information about availability of the booking option."""
        # Booking pointer
        # Do not consider booking two things in a single turn.
        vector = [0, 0]
        if turn_da.get('booking-nobook'):
            vector = [1, 0]
        if turn_da.get('booking-book') or turn_da.get('train-offerbooked'):
            vector = [0, 1]
        return vector

    def addDBPointer(self, domain, match_num, return_num=False):
        """Create database pointer for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        if domain in self.db_domains:
            vector = self.one_hot_vector(domain, match_num)
        else:
            vector = [0, 0, 0, 0]
        return vector

    def addDBIndicator(self, domain, match_num, return_num=False):
        """Create database indicator for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        if domain in self.db_domains:
            vector = self.one_hot_vector(domain, match_num)
        else:
            vector = [0, 0, 0, 0]

        # '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]'
        if vector == [0, 0, 0, 0]:
            indicator = '[db_null]'
        else:
            indicator = '[db_%s]' % vector.index(1)
        return indicator

    def get_match_num(self, constraints, return_entry=False):
        """Create database pointer for all related domains."""
        match = {'general': ''}
        entry = {}
        # if turn_domains is None:
        #     turn_domains = db_domains
        for domain in definitions.ALL_DOMAINS:
            match[domain] = ''
            if domain in self.db_domains and constraints.get(domain):
                matched_ents = self.queryJsons(domain, constraints[domain])
                match[domain] = len(matched_ents)
                if return_entry:
                    entry[domain] = matched_ents
        if return_entry:
            return entry
        return match

    def pointerBack(self, vector, domain):
        # multi domain implementation
        # domnum = cfg.domain_num
        if domain.endswith(']'):
            domain = domain[1:-1]
        if domain != 'train':
            nummap = {
                0: '0',
                1: '1',
                2: '2-3',
                3: '>3'
            }
        else:
            nummap = {
                0: '0',
                1: '1-5',
                2: '6-10',
                3: '>10'
            }
        if vector[:4] == [0, 0, 0, 0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain+': '+nummap[num] + '; '

        if vector[-2] == 0 and vector[-1] == 1:
            report += 'booking: ok'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'booking: unable'

        return report

    def queryJsons(self, domain, constraints, exactly_match=True, return_name=False):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state
        constraints: dict e.g. {'pricerange': 'cheap', 'area': 'west'}
        """
        # query the db
        if domain == 'taxi':
            return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']),
                     'taxi_types': random.choice(self.dbs[domain]['taxi_types']),
                     'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
        if domain == 'police':
            return self.dbs['police']
        if domain == 'hospital':
            if constraints.get('department'):
                for entry in self.dbs['hospital']:
                    if entry.get('department') == constraints.get('department'):
                        return [entry]
            else:
                return []

        valid_cons = False
        for v in constraints.values():
            if v not in ["not mentioned", "", "none"]:
                valid_cons = True
        if not valid_cons:
            return []

        match_result = []

        if 'name' in constraints:
            for db_ent in self.dbs[domain]:
                if 'name' in db_ent:
                    cons = constraints['name']
                    dbn = db_ent['name']
                    if cons == dbn:
                        db_ent = db_ent if not return_name else db_ent['name']
                        match_result.append(db_ent)
                        return match_result

        for db_ent in self.dbs[domain]:
            match = True
            for s, v in constraints.items():
                if v == "none":
                    continue

                if s == 'name':
                    continue
                if s in ['people', 'stay'] or (domain == 'hotel' and s == 'day') or \
                        (domain == 'restaurant' and s in ['day', 'time']):
                    continue

                skip_case = {"don't care": 1, "do n't care": 1,
                             "dont care": 1, "not mentioned": 1, "dontcare": 1, "": 1}
                if skip_case.get(v):
                    continue

                if s not in db_ent:
                    # logger.warning('Searching warning: slot %s not in %s db', s, domain)
                    match = False
                    break

                # v = 'guesthouse' if v == 'guest house' else v
                # v = 'swimmingpool' if v == 'swimming pool' else v
                v = 'yes' if v == 'free' else v

                if s in ['arrive', 'leave']:
                    try:
                        # raise error if time value is not xx:xx format
                        h, m = v.split(':')
                        v = int(h)*60+int(m)
                    except ValueError:
                        match = False
                        break
                    time = int(db_ent[s].split(':')[0])*60 + \
                        int(db_ent[s].split(':')[1])
                    if s == 'arrive' and v > time:
                        match = False
                    if s == 'leave' and v < time:
                        match = False
                else:
                    if exactly_match and v != db_ent[s]:
                        match = False
                        break
                    elif v not in db_ent[s]:
                        match = False
                        break

            if match:
                match_result.append(db_ent)

        if not return_name:
            return match_result
        else:
            if domain == 'train':
                match_result = [e['id'] for e in match_result]
            else:
                match_result = [e['name'] for e in match_result]
            return match_result
