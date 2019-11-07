/*********************************************
 * OPL 12.9.0.0 Model
 * Authors: istoleru + abucevschi
 * Creation Date: Oct 10, 2019 at 1:37:27 PM
 *********************************************/
 using CP;
 
 /* Number of items */
int m = ...;
/* Number of bidders */
int n = ...;
/* Number of offers*/
int l = ...;
/* Any subset belonging to items set */
tuple Bundle  
{
    {int} items;    
}
/* Any bid offer has a subset S and a price */
tuple BidOffer
{
    int bidOfferId;
    Bundle S;
    int price;
}

{BidOffer} offers = ...;

/* Bidder Id and BidOffer */
tuple Bidder
{
    int bidderId;
}

{Bidder} bidders = {<i> | i in 1..n};
{int} offerIDs = {i | i in 1..l};
{int} bidderIDs = {i | i in 1..n};

tuple Relation
{
    int bidOfferId;
    int bidderId;
}

{Relation} relations with bidOfferId in offerIDs, bidderId in bidderIDs = ...;
dvar boolean assignments[relations];

maximize sum(rel in relations, offer in offers : rel.bidOfferId == offer.bidOfferId) assignments[rel] * offer.price; 

subject to {
  /* Normalizarea */
  forall(el in relations) forall (offer in offers) if (offer.bidOfferId == el.bidOfferId && card(offer.S.items) == 0) offer.price == 0;   
   /* Monotonia */
  forall(el in relations) forall (offer in offers) forall(el2 in relations) forall (offer2 in offers)if (offer.bidOfferId == el.bidOfferId && offer2.bidOfferId == el2.bidOfferId && el != el2 && el.bidderId == el2.bidderId && card((offer.S.items inter offer2.S.items)) == card(offer.S.items)) offer.price <= offer2.price;
  /* Single assignment per object */
  forall(el in relations)  forall (offer in offers) forall(el2 in relations)  forall (offer2 in offers)if (offer.bidOfferId == el.bidOfferId && offer2.bidOfferId == el2.bidOfferId && el != el2 && card((offer.S.items inter offer2.S.items)) > 0) assignments[el] == 0 || assignments[el2] == 0; 
}