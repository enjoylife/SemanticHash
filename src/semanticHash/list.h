#ifndef Dblocks_List_h
#define Dblocks_List_h

#include <stdlib.h>
#include <stdbool.h>


typedef struct ListNode {
    struct ListNode *next;
    struct ListNode *prev;
    void *value;
} ListNode;

typedef struct List {
    int count;
    bool is_hollow;
    ListNode *first;
    ListNode *last;
} List;

List *List_create(bool is_data_internal);
List *List_copy(List *list);
List * List_slice(List *list, int l_index, int h_index);

void List_destroy(List *list);
void List_clear(List *list);
void List_clear_destroy(List *list);

void * List_search(List *list, void *item, int (*is_key)(void *, void *));
#define List_count(A) ((A)->count)
#define List_first(A) ((A)->first != NULL ? (A)->first->value : NULL)
#define List_last(A) ((A)->last != NULL ? (A)->last->value : NULL)

//void List_push(List *list, void *value);
void List_push(List *list, void *value);
void *List_pop(List *list);

void *List_remove(List *list, ListNode *node);

#define LIST_FOREACH(L, S, M, V) ListNode *_node = NULL;\
    ListNode *V = NULL;\
    for(V = _node = L->S; _node != NULL; V = _node = _node->M)

#endif
