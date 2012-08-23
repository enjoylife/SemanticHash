#include <stdarg.h>
#include <stdbool.h>
#include "list.h"
#include "dbg.h"



List *List_create(bool is_data_internal)
{
	List * new =  (List *) calloc(1, sizeof(List));
    new->is_hollow = is_data_internal;
    return new;
}

List *List_copy(List *old)
{
	//TODO Speed this up  by copying into a block?
	List *new = NULL;
	check_hard(old, "Must pass an initilized list");
	new = List_create(false);
	check_hard(new, "Failed to allocate memory");
	LIST_FOREACH(old, first, next, cur) {
		List_push(new, cur->value);
	}
   
	return new;
}

void List_destroy(List *list)
{
	check_hard(list, "Must pass an initilized list");
	LIST_FOREACH(list, first, next, cur) {
		if(cur->prev) {
			free(cur->prev);
		}
	}

	free(list->last);
	free(list);
}

void List_clear(List *list)
{

	check_hard(list, "Must pass an initilized list");
	LIST_FOREACH(list, first, next, cur) {
		free(cur->value);
	}
}

void List_clear_destroy(List *list)
{
	check_hard(list, "Must pass an initilized list");
	LIST_FOREACH(list, first, next, cur) {
		if(cur->prev) {
			free(cur->prev);
			free(cur->value);
		}
	}

	free(list->last);
	free(list);
}

void * List_search(List *list, void *item, int (*is_key)(void *, void *)){
    ListNode *node = list->first;
    while (node){
        if (is_key(node->value, item)){
            return node;
        }
        node = node->next;
    }
    return NULL;
}

void List_push(List *list, void *value)
{
	check_hard(list, "Must pass an initilized list");
	ListNode *node = calloc(1, sizeof(ListNode));
	check_mem(node);

	node->value = value;

	if(list->last == NULL) {
		list->first = node;
		list->last = node;
	} else {
		list->last->next = node;
		node->prev = list->last;
		list->last = node;
	}

	list->count++;

error:
	return;
}

void *List_pop(List *list)
{
	check_hard(list, "Must pass an initilized list");
	ListNode *node = list->last;
	return node != NULL ? List_remove(list, node) : NULL;
}

List *List_slice(List *list,int l_index, int h_index)
{
    int count = l_index;
	List * new = List_create(false);
	check_hard(list, "Must pass an initilized list");
	check_hard(l_index >=0 && h_index <= list->count && l_index < h_index , "index's are out of bounds");
	LIST_FOREACH(list, first, next, cur) {
        if(count<h_index){
            List_push(new,cur->value);
            count++;
        }
    }
    return new;
}
void *List_remove(List *list, ListNode *node)
{
	void *result = NULL;

	check_hard(list, "Must pass an initilized list");
	check(list->first && list->last, "List is empty.");
	check(node, "node can't be NULL");

	if(node == list->first && node == list->last) {
		list->first = NULL;
		list->last = NULL;
	} else if(node == list->first) {
		list->first = node->next;
	} else if (node == list->last) {
		list->last = node->prev;
	} else {
		ListNode *after = node->next;
		ListNode *before = node->prev;
		after->prev = before;
		before->next = after;
	}

	list->count--;
	result = node->value;
	free(node);

error:
	return result;
}

