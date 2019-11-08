#include <bits/stdc++.h>

#define pb push_back
#define mp make_pair
#define mt make_tuple
#define ll long long
#define pii pair<int,int>
#define tii tuple <int,int,int>
#define N 100005
#define mod 2000003
#define X first
#define Y second
#define eps 0.0000000001
#define all(x) x.begin(),x.end()
#define tot(x) x+1,x+n+1
using namespace std;

const int dx[] = {0, 1, 0, -1};
const int dy[] = {1, 0, -1, 0};

struct offer {
    ll bidOfferId;
    ll price;
    ll mask;
    //set<int>bundle;

};
struct relation {
    ll bidOfferId;
    ll bidderId;
};

ll n, m, l, k, i, x;
string string_mask;
ll sum, sol, sol_mask, mask, mm, j, pos, offers_mask, sol_offers_mask;
bool ok;
int main() {
    //cin.sync_with_stdio(0);
    // cout.sync_with_stdio(0);
    freopen("data.in", "r", stdin);
    scanf("%lld%lld%lld", &n, &m, &l); // citim numarul de bidders, de items si numarul de oferte
    offer offers[l];
    relation relations[l];

    for(i = 1; i <= l; i++) {
        //citim oferta i
        scanf("%lld", &offers[i].bidOfferId); // id-ul ofertei
        scanf("%lld", &k); // numarul de elemente din oferta i
        mask = 0;

        //citim bundle-ul si cream masca
        for(; k; k--) {
            scanf("%lld", &x);
            offers[i].mask |= (1ll << x);
            //offers[i].bundle.insert(x);
        }

        scanf("%lld", &offers[i].price);
    }

    //citim relatiile
    for(i = 1; i <= l; i++)
        scanf("%lld%lld", &relations[i].bidOfferId, &relations[i].bidderId);

    for(i = 1; i <= l; i++) {
        if(offers[i].mask == 0 && offers[i].price > 0) {
            cout << "Nu este respectata normalizarea\n";
            return 0;
        }

        for(j = i + 1; j <= l; j++) {
            int bidOffer1 = relations[i].bidOfferId;
            int bidOffer2 = relations[j].bidOfferId;

            if(relations[i].bidderId == relations[j].bidderId) {
                if((offers[bidOffer1].mask & offers[bidOffer2].mask) == offers[bidOffer1].mask) // primul inclus in al doilea
                    if(offers[bidOffer1].price > offers[bidOffer2].price) {
                        cout << "Nu este respectata monotonia\n";
                        return 0;
                    }

                if((offers[bidOffer1].mask & offers[bidOffer2].mask) == offers[bidOffer2].mask) // al doilea inclus in primul
                    if(offers[bidOffer2].price > offers[bidOffer1].price) {
                        cout << "Nu este respectata monotonia\n";
                        return 0;
                    }
            }
        }
    }

    sol = 0;
    mm = (1ll << l);

    for(i = 1; i < mm; i++) {
        x = i;
        pos = 1;
        mask = 0;
        sum = 0;
        ok = 1;
        offers_mask = 0;

        while(x) {
            if(x & 1) {
                if(mask & offers[pos].mask) {
                    ok = 0;
                    break; // Single assignment per object
                }

                sum += offers[pos].price;
                mask |= offers[pos].mask;
                offers_mask |= (1ll << pos);

            }

            x >>= 1;
            pos++;
        }

        if(sum > sol) {
            //cout << "*" << i;
            sol = sum;
            sol_mask = mask;
            //sol_offers_mask = offers_mask;
            sol_offers_mask = i;
        }
    }

    for(i = 1; i <= l; i++)
        string_mask += "0";

    pos = 0;

    while(sol_offers_mask) {
        if(sol_offers_mask & 1)
            string_mask[pos] = '1';

        pos++;
        sol_offers_mask >>= 1;
    }

    //reverse(all(string_mask));
    printf("maximum sum: %lld assignment mask: %s", sol, string_mask.c_str());
    return 0;
}
